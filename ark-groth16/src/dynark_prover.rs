#[cfg(feature = "parallel")]
use crate::{
    data_structures::Cache,
    fft_handler::{
        convolution_f_f, convolution_f_g, elementwise_add_g_g, elementwise_product_f_g,
    },
    r1cs_to_qap::R1CSToQAP,
    utils::{
        inner_product_sparse_f_dense_g, lagrange_of_dense_vec,
        improved_lagrange_sparse_get
    },
    Groth16, Proof, ProvingKey, UpdatingKey,
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{
    ConstraintMatrices, SynthesisError,
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
    ///
    #[inline]
    pub fn compute_projection_quotient(
        uk: &UpdatingKey<E>,
        domain: D<E::ScalarField>,
        a: &[E::ScalarField],
        b: &[E::ScalarField],
        c: &[E::ScalarField],
    ) -> (Vec<E::G1>, Vec<E::G1>) {
        let n = domain.size();
        debug_assert_eq!(a.len(), n);
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(c.len(), n);

        let lagrg_trunc = uk.lagrange_truncated.clone();
        let w = &uk.w;
        let w_rev = &uk.w_reverse;
        let p = uk.p.clone();
        

        let a_l_w = convolution_f_g::<E>(w, &elementwise_product_f_g::<E>(a, &lagrg_trunc));
        let a_w_rev_l =
            elementwise_product_f_g::<E>(&convolution_f_f::<E>(a, w_rev), &lagrg_trunc);
        let q_a_first = elementwise_add_g_g::<E>(&a_l_w, &a_w_rev_l);

        let q_a_second = elementwise_product_f_g::<E>(&a, &p);
        let q_a = elementwise_add_g_g::<E>(&q_a_first, &q_a_second);

        let b_l_w = convolution_f_g::<E>(&w, &elementwise_product_f_g::<E>(b, &lagrg_trunc));
        let b_w_rev_l =
            elementwise_product_f_g::<E>(&convolution_f_f::<E>(b, &w_rev), &lagrg_trunc);

        let q_b_first = elementwise_add_g_g::<E>(&b_l_w, &b_w_rev_l);

        let q_b_second = elementwise_product_f_g::<E>(b, &p);
        let q_b = elementwise_add_g_g::<E>(&q_b_first, &q_b_second);


        (q_a, q_b)
    }

    /// Proof raw a,b,c without random r,s
    pub fn calc_raw_proof(
        pk: &ProvingKey<E>,
        domain: &GeneralEvaluationDomain<<E>::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        a: &Vec<E::ScalarField>,
        b: &Vec<E::ScalarField>,
        c: &Vec<E::ScalarField>
    ) -> (E::G1, E::G1, E::G2, E::G1){
        // println!("A_Lagrange: {:?}", a);
        // println!("B_Lagrange: {:?}", b);
        // println!("C_Lagrange: {:?}", c);
        
        let quotient = QAP::compute_quotient::<E::ScalarField, D<E::ScalarField>>(
            &domain,
            a.clone(),
            b.clone(),
            c.clone(),
        ).unwrap();
        
        let r = E::ScalarField::zero();
        let s = E::ScalarField::zero();

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
        let a_g1 = r_delta_g1 +
            pk.vk.alpha_g1 + E::G1::msm_bigint(&pk.u_g1, &assignment);

        // Compute B in G1
        let s_delta_g1 = pk.delta_g1.mul(s);
        let b_g1 = s_delta_g1 + pk.beta_g1 + E::G1::msm_bigint(&pk.v_g1, &assignment);

        // Compute B in G2
        let s_delta_g2 = pk.vk.delta_g2.mul(s);
        let b_g2 = s_delta_g2 + pk.vk.beta_g2 + E::G2::msm_bigint(&pk.v_g2, &assignment);

        // Compute C in G1
        let c_g1 = a_g1 * &s + b_g1 * &r - &r_s_delta_g1 + &t_p_witness_g1 + &q_g1;

        (a_g1,b_g1,b_g2,c_g1)


    }

    /// Generates a proof of satisfaction of the arithmetic circuit C (specified
    /// as R1CS constraints).
    pub fn prove_dynark<R: Rng>(
        pk: &ProvingKey<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        rng: &mut R,
    ) -> Result<(Proof<E>, Cache<E>), SynthesisError> {
        let num_constraints = r1cs.num_constraints;
        let domain_min_size = num_constraints;
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(domain_min_size)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();

        let (a, b, c) = lagrange_of_dense_vec(&r1cs, &[instance, witness].concat()).unwrap();

        // println!("A_Lagrange: {:?}", a);
        // println!("B_Lagrange: {:?}", b);
        // println!("C_Lagrange: {:?}", c);
        
        let quotient = QAP::compute_quotient::<E::ScalarField, D<E::ScalarField>>(
            &domain,
            a.clone(),
            b.clone(),
            c.clone(),
        )?;
        

        let r = E::ScalarField::rand(rng);
        let s = E::ScalarField::rand(rng);
        // let r = E::ScalarField::zero();
        // let s = E::ScalarField::zero();

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
        let a_g1 = r_delta_g1 +
            pk.vk.alpha_g1 + E::G1::msm_bigint(&pk.u_g1, &assignment);

        // Compute B in G1
        let s_delta_g1 = pk.delta_g1.mul(s);
        let b_g1 = s_delta_g1 + pk.beta_g1 + E::G1::msm_bigint(&pk.v_g1, &assignment);

        // Compute B in G2
        let s_delta_g2 = pk.vk.delta_g2.mul(s);
        let b_g2 = s_delta_g2 + pk.vk.beta_g2 + E::G2::msm_bigint(&pk.v_g2, &assignment);

        // Compute C in G1
        let c_g1 = a_g1 * &s + b_g1 * &r - &r_s_delta_g1 + &t_p_witness_g1 + &q_g1;

        let proof = Proof {
            a_g1: a_g1.into_affine(),
            b_g2: b_g2.into_affine(),
            c_g1: c_g1.into_affine(),
        };

        let cache = Cache {
            domain,
            num_instance: instance.len(),
            a_g1: proof.a_g1,
            b_g1: b_g1.into_affine(),
            b_g2: proof.b_g2,
            c_g1: proof.c_g1,
            r,
            s,
            q_a: Default::default(),
            q_b: Default::default(),
        };

        Ok((proof, cache))
    }

    /// Preprocess
    pub fn process_dynark(
        uk: &UpdatingKey<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        cache: &mut Cache<E>,
    ) -> Result<(), SynthesisError> {
        let (a, b, c) = lagrange_of_dense_vec(&r1cs, &[instance, witness].concat()).unwrap();
        let (q_a, q_b) = Self::compute_projection_quotient(uk, cache.domain, &a, &b, &c);
        cache.q_a = q_a;
        cache.q_b = q_b;
        Ok(())
    }

    /// Update
    /// compute a_acc' = a_acc + (r-r') * delta_g1 + \sum update[i] * u[i]
    /// compute b_acc' = b_acc + (s-s') * delta_g2 + \sum update[i] * v[i]
    /// compute c_acc' = c_acc + s' * a_acc' - s * a_acc + r' * b_acc' - r *
    /// b_acc  - (r's' - rs) * delta_g1 + \sum witness_update[i] * t_p_g1[i]
    /// Updates the proof with new instance and witness values.
    pub fn update_dynark(
        uk: &UpdatingKey<E>,
        matrices: &ConstraintMatrices<E::ScalarField>,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
        cache: &Cache<E>,
    ) -> Result<Proof<E>, SynthesisError> {
        let n = cache.domain.size();
        
        // println!("Instance Update & Witness Update ::: {:?} || {:?}", instance_update, witness_update);
        let update = [&instance_update[..], &witness_update[..]].concat();
        let r_new = E::ScalarField::rand(&mut ark_std::test_rng());
        let s_new = E::ScalarField::rand(&mut ark_std::test_rng());
        
        let delta_g1 = uk.pk.delta_g1;
        let delta_g2 = uk.pk.vk.delta_g2;
        
        let mut update_c_g1_linear_combination: E::G1 = witness_update
            .par_iter()
            .map(|(i, update_i)| uk.pk.t_p_g1[*i - cache.num_instance].into_group() * update_i)
            .sum();
    
    
        let update_a_linear_combine: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.u_g1[*i].into_group() * update_i)
            .sum();
        let a_g1_new =
            cache.a_g1.into_group() + delta_g1 * (r_new - cache.r) + update_a_linear_combine;

        let update_b_g1_linear_combine: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.v_g1[*i].into_group() * update_i)
            .sum();
        let b_g1_new =
            cache.b_g1.into_group() + delta_g1 * (s_new - cache.s) + update_b_g1_linear_combine;

        let update_b_g2_linear_combine: E::G2 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.v_g2[*i].into_group() * update_i)
            .sum();
        let b_g2_new =
            cache.b_g2.into_group() + delta_g2 * (s_new -cache.s) + update_b_g2_linear_combine;

        update_c_g1_linear_combination += a_g1_new * &s_new - cache.a_g1 * &cache.s
            + b_g1_new * &r_new
            - cache.b_g1 * &cache.r
            - delta_g1 * (r_new * &s_new - cache.r * &cache.s);

        let (a_lagrange_sparse, b_lagrange_sparse) =
            improved_lagrange_sparse_get(&uk.a_matrix_cols, &uk.b_matrix_cols, &update).unwrap();

        let c_project = inner_product_sparse_f_dense_g::<E>(&a_lagrange_sparse, &cache.q_b)
            + inner_product_sparse_f_dense_g::<E>(&b_lagrange_sparse, &cache.q_a);

        // println!("Cross A({:?}) B({:?})", a_lagrange_sparse.len(), b_lagrange_sparse.len());
        let c_cross_1 = a_lagrange_sparse.par_iter().map(|(i,a)|{
            let ot_sum = b_lagrange_sparse.par_iter().map(|(j,b)|{
                if i==j {E::ScalarField::zero()} else {uk.w[(n + j - i) % n]*b} 
            }).sum::<E::ScalarField>();
            uk.lagrange_truncated[*i] * (*a * ot_sum)
        }).sum::<E::G1>();

        let c_cross_2 = b_lagrange_sparse.par_iter().map(|(j,b)|{
            let ot_sum = a_lagrange_sparse.par_iter().map(|(i,a)|{
                if i==j {E::ScalarField::zero()} else {uk.w[(n + *i - *j) % n]*a} 
            }).sum::<E::ScalarField>();
            uk.lagrange_truncated[*j] * (*b * ot_sum)
        }).sum::<E::G1>();

        
        let mut results = Vec::<(usize, E::ScalarField)>::new();
        let mut cur_i = 0;
        let mut cur_j = 0;
        let mut a_copy = a_lagrange_sparse;
        let mut b_copy = b_lagrange_sparse;
        a_copy.sort_by_key(|x| x.0);
        b_copy.sort_by_key(|y| y.0);

        // two-pointer merge join
        while cur_i < a_copy.len() && cur_j < b_copy.len() {
            match a_copy[cur_i].0.cmp(&b_copy[cur_j].0) {
                std::cmp::Ordering::Less => cur_i += 1,
                std::cmp::Ordering::Greater => cur_j += 1,
                std::cmp::Ordering::Equal => {
                    results.push((a_copy[cur_i].0, a_copy[cur_i].1.clone() * b_copy[cur_j].1.clone()));
                    cur_i += 1;
                    cur_j += 1;
                }
            }
        }
        let c_cross_3 = results.par_iter().map(|(i,val)| uk.p[*i] * val).sum::<E::G1>();

        let c_new = cache.c_g1.into_group() + update_c_g1_linear_combination + c_project + c_cross_1 + c_cross_2 + c_cross_3;

        Ok(Proof {
            a_g1: a_g1_new.into_affine(),
            b_g2: b_g2_new.into_affine(),
            c_g1: c_new.into_affine(),
        })
    }
}
