#[cfg(feature = "parallel")]
use crate::{
    r1cs_to_qap::R1CSToQAP,
    Groth16, PreparedVerifyingKey, Proof,
};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::PrimeField;
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{
    Result as R1CSResult, SynthesisError,
};
use ark_std::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type D<F> = GeneralEvaluationDomain<F>;

#[cfg(feature = "parallel")]
impl<E: Pairing, QAP: R1CSToQAP> Groth16<E, QAP> {
    /// Verify a Groth16 proof `proof` against the prepared verification key
    /// `pvk`, with respect to the instance `public_inputs`.
    pub fn verify_dynark(
        pvk: &PreparedVerifyingKey<E>,
        proof: &Proof<E>,
        instance: &[E::ScalarField],
    ) -> R1CSResult<bool> {
        if instance.len() != pvk.vk.t_v_g1.len() {
            return Err(SynthesisError::MalformedVerifyingKey);
        }

        let instance = instance
            .par_iter()
            .map(|instance_i| instance_i.into_bigint())
            .collect::<Vec<_>>();

        let t_v_instance_g1 = E::G1::msm_bigint(&pvk.vk.t_v_g1, &instance);

        let qap = E::multi_miller_loop(
            [
                <E::G1Affine as Into<E::G1Prepared>>::into(proof.a_g1),
                t_v_instance_g1.into_affine().into(),
                proof.c_g1.into(),
            ],
            [
                proof.b_g2.into(),
                pvk.gamma_g2_neg_pc.clone(),
                pvk.delta_g2_neg_pc.clone(),
            ],
        );

        let test = E::final_exponentiation(qap).ok_or(SynthesisError::UnexpectedIdentity)?;

        Ok(test.0 == pvk.alpha_g1_beta_g2)
    }
}
