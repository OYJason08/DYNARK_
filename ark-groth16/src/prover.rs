#[cfg(feature = "parallel")]
use crate::{
    r1cs_to_qap::R1CSToQAP,
    Groth16, Proof, ProvingKey, VerifyingKey,
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, PrimeField, UniformRand, Zero};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{
    ConstraintMatrices, ConstraintSynthesizer, ConstraintSystem, OptimizationGoal,
    Result as R1CSResult,
};
use ark_std::{
    cfg_into_iter, cfg_iter,
    ops::Mul,
    rand::Rng,
    vec::Vec,
};
use itertools::Itertools;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type D<F> = GeneralEvaluationDomain<F>;

#[cfg(feature = "parallel")]
impl<E: Pairing, QAP: R1CSToQAP> Groth16<E, QAP> {
    /// Create a Groth16 proof using randomness `r` and `s` and
    /// the provided R1CS-to-QAP reduction, using the provided
    /// R1CS constraint matrices.

    ///
    #[inline]
    pub fn generate_proof_with_matrices(
        pk: &ProvingKey<E>,
        r: E::ScalarField,
        s: E::ScalarField,
        matrices: &ConstraintMatrices<E::ScalarField>,
        num_inputs: usize,
        num_constraints: usize,
        full_assignment: &[E::ScalarField],
    ) -> R1CSResult<Proof<E>> {
        let prover_time = start_timer!(|| "Groth16::Prover");
        let witness_map_time = start_timer!(|| "R1CS to QAP witness map");
        let h = QAP::compute_quotient_from_matrices::<E::ScalarField, D<E::ScalarField>>(
            matrices,
            num_inputs,
            num_constraints,
            full_assignment,
        )?;
        end_timer!(witness_map_time);
        let input_assignment = &full_assignment[1..num_inputs];
        let aux_assignment = &full_assignment[num_inputs..];
        let proof =
            Self::create_proof_with_assignment(pk, r, s, &h, input_assignment, aux_assignment)?;
        end_timer!(prover_time);

        Ok(proof)
    }

    /// Create a Groth16 proof that is zero-knowledge using the provided
    /// R1CS-to-QAP reduction.
    /// This method samples randomness for zero knowledges via `rng`.
    #[inline]
    pub fn generate_proof_with_circuit<C>(
        circuit: C,
        pk: &ProvingKey<E>,
        rng: &mut impl Rng,
    ) -> R1CSResult<Proof<E>>
    where
        C: ConstraintSynthesizer<E::ScalarField>,
    {
        Self::create_proof_with_reduction(
            circuit,
            pk,
            E::ScalarField::rand(rng),
            E::ScalarField::rand(rng),
        )
    }

    /// Create a Groth16 proof using randomness `r` and `s` and the provided
    /// R1CS-to-QAP reduction.
    #[inline]
    pub fn create_proof_with_reduction<C>(
        circuit: C,
        pk: &ProvingKey<E>,
        r: E::ScalarField,
        s: E::ScalarField,
    ) -> R1CSResult<Proof<E>>
    where
        E: Pairing,
        C: ConstraintSynthesizer<E::ScalarField>,
        QAP: R1CSToQAP,
    {
        let prover_time = start_timer!(|| "Groth16::Prover");
        let cs = ConstraintSystem::new_ref();

        // Set the optimization goal
        cs.set_optimization_goal(OptimizationGoal::Constraints);

        // Synthesize the circuit.
        let synthesis_time = start_timer!(|| "Constraint synthesis");
        circuit.generate_constraints(cs.clone())?;
        debug_assert!(cs.is_satisfied().unwrap());
        end_timer!(synthesis_time);

        let lc_time = start_timer!(|| "Inlining LCs");
        cs.finalize();
        end_timer!(lc_time);

        let (domain, a, b, c) =
            QAP::lagrange_form_from_cs::<E::ScalarField, D<E::ScalarField>>(cs.clone()).unwrap();

        let compute_quotient_time = start_timer!(|| "R1CS to QAP witness map");
        let q = QAP::compute_quotient::<E::ScalarField, D<E::ScalarField>>(
            &domain,
            a.clone(),
            b.clone(),
            c.clone(),
        )?;
        end_timer!(compute_quotient_time);

        let prover = cs.borrow().unwrap();
        let proof = Self::create_proof_with_assignment(
            pk,
            r,
            s,
            &q,
            &prover.instance_assignment,
            &prover.witness_assignment,
        )?;

        end_timer!(prover_time);

        Ok(proof)
    }

    ///
    #[inline]
    pub fn create_proof_with_assignment(
        pk: &ProvingKey<E>,
        r: E::ScalarField,
        s: E::ScalarField,
        quotient: &[E::ScalarField],
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
    ) -> R1CSResult<Proof<E>> {
        let quotient = cfg_into_iter!(quotient)
            .map(|q_i| q_i.into_bigint())
            .collect::<Vec<_>>();
        let q_g1 = E::G1::msm_bigint(&pk.tau_powers_g1, &quotient);

        let instance = instance
            .iter()
            .map(|instance_i| instance_i.into_bigint())
            .collect::<Vec<_>>();

        let witness = cfg_iter!(witness)
            .map(|witness_i| witness_i.into_bigint())
            .collect::<Vec<_>>();

        let t_p_witness_g1 = E::G1::msm_bigint(&pk.t_p_g1, &witness);

        let r_s_delta_g1 = pk.delta_g1 * (r * s);

        let assignment = [&instance[..], &witness[..]].concat();

        // Compute A in G1
        let r_delta_g1 = pk.delta_g1.mul(r);
        let a_g1 = r_delta_g1 + pk.vk.alpha_g1 + E::G1::msm_bigint(&pk.u_g1, &assignment);

        // Compute B in G1
        let s_delta_g1 = pk.delta_g1.mul(s);
        let b_g1 = s_delta_g1 + pk.beta_g1 + E::G1::msm_bigint(&pk.v_g1, &assignment);

        // Compute B in G2
        let s_delta_g2 = pk.vk.delta_g2.mul(s);
        let b_g2 = s_delta_g2 + pk.vk.beta_g2 + E::G2::msm_bigint(&pk.v_g2, &assignment);

        // Compute C in G1
        let c_g1 = a_g1 * &s + b_g1 * &r - &r_s_delta_g1 + &t_p_witness_g1 + &q_g1;

        Ok(Proof {
            a_g1: a_g1.into_affine(),
            b_g2: b_g2.into_affine(),
            c_g1: c_g1.into_affine(),
        })
    }

    /// Given a Groth16 proof, returns a fresh proof of the same statement. For
    /// a proof π of a statement S, the output of the non-deterministic
    /// procedure `rerandomize_proof(π)` is statistically indistinguishable
    /// from a fresh honest proof of S. For more info, see theorem 3 of [\[BKSV20\]](https://eprint.iacr.org/2020/811)
    pub fn rerandomize_proof(
        vk: &VerifyingKey<E>,
        proof: &Proof<E>,
        rng: &mut impl Rng,
    ) -> Proof<E> {
        // These are our rerandomization factors. They must be nonzero and uniformly
        // sampled.
        let (mut r1, mut r2) = (E::ScalarField::zero(), E::ScalarField::zero());
        while r1.is_zero() || r2.is_zero() {
            r1 = E::ScalarField::rand(rng);
            r2 = E::ScalarField::rand(rng);
        }

        // See figure 1 in the paper referenced above:
        //   A' = (1/r₁)A
        //   B' = r₁B + r₁r₂(δG₂)
        //   C' = C + r₂A

        // We can unwrap() this because r₁ is guaranteed to be nonzero
        let new_a = proof.a_g1.mul(r1.inverse().unwrap());
        let new_b = proof.b_g2.mul(r1) + &vk.delta_g2.mul(r1 * &r2);
        let new_c = proof.c_g1 + proof.a_g1.mul(r2).into_affine();

        Proof {
            a_g1: new_a.into_affine(),
            b_g2: new_b.into_affine(),
            c_g1: new_c.into_affine(),
        }
    }

    // fn calculate_coeff<G: AffineRepr>(
    //     mask_0: G::Group,
    //     u_or_v: &[G],
    //     mask_1: G,
    //     assignment_without1: &[<G::ScalarField as PrimeField>::BigInt],
    // ) -> G::Group
    // where
    //     G::Group: VariableBaseMSM<MulBase = G>,
    // {
    //     //let el = u_or_v[0];
    //     let acc = G::Group::msm_bigint(&u_or_v, assignment_without1);

    //     let mut res = mask_0;
    //     //res.add_assign(&el);
    //     res += &acc;
    //     res.add_assign(&mask_1);
    //     res
    // }

    // Create a Groth16 proof that is *not* zero-knowledge with the provided
    // R1CS-to-QAP reduction.
    // #[inline]
    // pub fn create_proof_with_reduction_no_zk<C>(
    //     circuit: C,
    //     pk: &ProvingKey<E>,
    // ) -> R1CSResult<Proof<E>>
    // where
    //     C: ConstraintSynthesizer<E::ScalarField>,
    // {
    //     Self::create_proof_with_reduction(
    //         circuit,
    //         pk,
    //         E::ScalarField::zero(),
    //         E::ScalarField::zero(),
    //     )
    // }
}
