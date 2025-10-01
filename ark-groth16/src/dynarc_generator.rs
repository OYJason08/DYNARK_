use crate::{
    fft_handler::convolution_f_g,
    r1cs_to_qap::{tau_powers_field_lifted, R1CSToQAP},
    Groth16, ProvingKey, UpdatingKey, Vec, VerifyingKey,
};
use ark_ec::{pairing::Pairing, scalar_mul::BatchMulPreprocessing, AffineRepr, CurveGroup};
use ark_ff::{Field, PrimeField, UniformRand, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{
    ConstraintMatrices,
    Result as R1CSResult, SynthesisError,
};
use ark_std::rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl<E: Pairing, QAP: R1CSToQAP> Groth16<E, QAP> {
    ///
    pub fn groth16_setup_dynarc<R: Rng>(
        r1cs: ConstraintMatrices<E::ScalarField>,
        rng: &mut R,
    ) -> Result<(ProvingKey<E>, VerifyingKey<E>, GeneralEvaluationDomain::<E::ScalarField>), SynthesisError> {

        let domain_min_size =  r1cs.num_constraints;
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(domain_min_size)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let tau = domain.sample_element_outside_domain(rng);
        let alpha = E::ScalarField::rand(rng);
        let beta = E::ScalarField::rand(rng);
        let gamma = E::ScalarField::rand(rng);
        let delta = E::ScalarField::rand(rng);

        let g1_gen = E::G1::rand(rng).into_affine();
        let g2_gen = E::G2::rand(rng).into_affine();

        let pk = Self::generate_prover_key_with_matrices(
            domain,
            &r1cs,
            (tau, alpha, beta, gamma, delta, g1_gen, g2_gen),
        )?;

        let vk = pk.vk.clone();

        // let uk = Self::generate_updating_keys(&r1cs, &domain, &pk)?;

        Ok((pk, vk, domain))
    }

    ///
    pub fn generate_updating_keys(
        r1cs: &ConstraintMatrices<E::ScalarField>,
        domain: &GeneralEvaluationDomain<E::ScalarField>,
        pk: &ProvingKey<E>,
    ) -> R1CSResult<UpdatingKey<E>> {
        let domain_size = domain.size();
        debug_assert!((domain_size % 2) == 0);

        let mut tau_powers_truncated = pk.tau_powers_g1.clone();
        tau_powers_truncated.push(E::G1Affine::zero());
        let tau_powers_truncated: Vec<<E as Pairing>::G1> = tau_powers_truncated
            .par_iter()
            .map(|x| (*x).into())
            .collect();

        let lagrange_truncated = domain.ifft(&tau_powers_truncated);

        let domain_gen: E::ScalarField = domain.group_gen();

        let one_over_n = E::ScalarField::from(domain_size as u64).inverse().unwrap();
        // let time_t = Instant::now();
        let w: Vec<E::ScalarField> = (0..domain_size)
            .scan(domain_gen.inverse().unwrap(), |state, _| {
                *state *= domain_gen;
                Some(*state)
            })
            .map(|gi| {
                let denom = E::ScalarField::ONE - gi;
                if denom.is_zero() {
                    E::ScalarField::zero()
                } else {
                    gi * denom.inverse().unwrap() * one_over_n
                }
            })
            .collect();


        let w_reverse: Vec<E::ScalarField> = (0..domain_size).into_par_iter().map(|i|{
            if i ==0 {
                E::ScalarField::zero()
            }
            else{
                w[domain_size - i]
            }
        }).collect();

        let coeff1 = w.iter().fold(E::ScalarField::zero(), |acc, x| acc + x);
        let conv: Vec<E::G1> = convolution_f_g::<E>(&w, &lagrange_truncated);

        let p: Vec<E::G1> = lagrange_truncated
            .par_iter()
            .zip(conv.par_iter())
            .map(|(x, y)| -(*x * coeff1 + *y))
            .collect();


        debug_assert_eq!(lagrange_truncated.len(), domain_size);
        debug_assert_eq!(w.len(), domain_size);
        debug_assert_eq!(w_reverse.len(), domain_size);
        debug_assert_eq!(p.len(), domain_size);
        
        let w_for_q = domain.fft(&w);
        let w_reverse_for_q = domain.fft(&w_reverse);

        let mut a_matrix_cols: Vec<Vec<(usize,E::ScalarField)>> = vec![vec![]; domain_size];
        let mut b_matrix_cols: Vec<Vec<(usize,E::ScalarField)>> = vec![vec![]; domain_size];

        (r1cs.a).iter().enumerate()
        .for_each(|(i, a_row)| {
            a_row.iter().for_each(|(val, j)|{a_matrix_cols[*j].push((i,*val))});
        });

        (r1cs.b).iter().enumerate()
        .for_each(|(i, b_row)| {
            b_row.iter().for_each(|(val, j)|{b_matrix_cols[*j].push((i,*val))});
        });

        Ok(UpdatingKey {
            pk: pk.clone(),
            lagrange_truncated,
            w,
            w_reverse,
            p,
            w_for_q,
            w_reverse_for_q,
            a_matrix_cols,
            b_matrix_cols
        })
    }

    /// Create parameters for a circuit, given some toxic waste, R1CS to QAP
    /// calculator and group generators
    pub fn generate_prover_key_with_matrices(
        domain: GeneralEvaluationDomain<E::ScalarField>,
        matrices: &ConstraintMatrices<E::ScalarField>,
        randomness: (
            E::ScalarField,
            E::ScalarField,
            E::ScalarField,
            E::ScalarField,
            E::ScalarField,
            E::G1Affine,
            E::G2Affine,
        ),
    ) -> R1CSResult<ProvingKey<E>> {
        type D<F> = GeneralEvaluationDomain<F>;
        let (tau, alpha, beta, gamma, delta, g1_gen, g2_gen) = randomness;
        let g1_gen = g1_gen.into_group();
        let g2_gen = g2_gen.into_group();

        let num_constraints = matrices.num_constraints;
        let num_instance = matrices.num_instance_variables;
        let num_witness = matrices.num_witness_variables;
        let domain_size = domain.size();
        let qap_num_variables = num_instance + num_witness;

        // process u, v, w
        ///////////////////////////////////////////////////////////////////////////

        let (u, v, w, vanish_tau) = Self::compute_u_v_w::<E::ScalarField, D<E::ScalarField>>(
            &domain,
            num_constraints,
            num_instance,
            num_witness,
            matrices,
            &tau,
        )?;

        // Compute query densities
        let non_zero_u: usize = cfg_into_iter!(0..(qap_num_variables)) // why qap_num_variables not qap_num_variables + 1
            .map(|i| usize::from(!u[i].is_zero()))
            .sum();

        let non_zero_v: usize = cfg_into_iter!(0..(qap_num_variables))
            .map(|i| usize::from(!v[i].is_zero()))
            .sum();

        // process field elements
        ///////////////////////////////////////////////////////////////////////////

        let gamma_inverse = gamma.inverse().ok_or(SynthesisError::UnexpectedIdentity)?;
        let delta_inverse = delta.inverse().ok_or(SynthesisError::UnexpectedIdentity)?;

        let t_v = cfg_iter!(u[..num_instance])
            .zip(&v[..num_instance])
            .zip(&w[..num_instance])
            .map(|((u_j, v_j), w_j)| (beta * u_j + &(alpha * v_j) + w_j) * &gamma_inverse)
            .collect::<Vec<_>>();

        let t_p = cfg_iter!(u[num_instance..])
            .zip(&v[num_instance..])
            .zip(&w[num_instance..])
            .map(|((u_j, v_j), w_j)| (beta * u_j + &(alpha * v_j) + w_j) * &delta_inverse)
            .collect::<Vec<_>>();

        let tau_powers = tau_powers_field_lifted::<_, D<E::ScalarField>>(
            domain_size - 1,
            tau,
            vanish_tau,
            delta_inverse,
        )?;

        // process group elements
        ///////////////////////////////////////////////////////////////////////////

        // Compute G window table
        let g1_table = BatchMulPreprocessing::new(
            g1_gen,
            non_zero_u + non_zero_v + qap_num_variables + domain_size,
        );
        let g2_table = BatchMulPreprocessing::new(g2_gen, non_zero_v);

        let alpha_g1 = g1_gen * &alpha;
        let beta_g1 = g1_gen * &beta;
        let beta_g2 = g2_gen * &beta;
        let delta_g1 = g1_gen * &delta;
        let delta_g2 = g2_gen * &delta;

        let u_g1 = g1_table.batch_mul(&u);

        let v_g1 = g1_table.batch_mul(&v);

        let v_g2 = g2_table.batch_mul(&v);

        let t_p_g1 = g1_table.batch_mul(&t_p);

        let t_v_g1 = g1_table.batch_mul(&t_v);

        let tau_powers_g1 = g1_table.batch_mul(&tau_powers);

        let gamma_g2 = g2_gen * &gamma;

        let vk = VerifyingKey::<E> {
            num_constraints,
            num_instance,
            num_witness,
            domain,
            alpha_g1: alpha_g1.into_affine(),
            beta_g2: beta_g2.into_affine(),
            gamma_g2: gamma_g2.into_affine(),
            delta_g2: delta_g2.into_affine(),
            t_v_g1,
        };

        Ok(ProvingKey {
            vk,
            beta_g1: beta_g1.into_affine(),
            delta_g1: delta_g1.into_affine(),
            u_g1,
            v_g1,
            v_g2,
            tau_powers_g1,
            t_p_g1,
        })
    }

    ///
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn compute_u_v_w<F: PrimeField, D: EvaluationDomain<F>>(
        domain: &D,
        num_constraints: usize,
        num_instance_variables: usize,
        num_witness_variables: usize,
        matrices: &ConstraintMatrices<F>,
        tau: &F,
    ) -> Result<(Vec<F>, Vec<F>, Vec<F>, F), SynthesisError> {
        debug_assert!(matrices.num_constraints == num_constraints);
        debug_assert!(matrices.num_instance_variables == num_instance_variables);
        debug_assert!(matrices.num_witness_variables == num_witness_variables);

        let vanish_tau = domain.evaluate_vanishing_polynomial(*tau);

        // Evaluate all Lagrange polynomials
        let coefficients_time = start_timer!(|| "Evaluate Lagrange coefficients");
        let lagrange_vec_tau: Vec<F> = domain.evaluate_all_lagrange_coefficients(*tau);
        end_timer!(coefficients_time);

        let mut u: Vec<F> = vec![F::zero(); num_instance_variables + num_witness_variables];
        let mut v: Vec<F> = vec![F::zero(); num_instance_variables + num_witness_variables];
        let mut w: Vec<F> = vec![F::zero(); num_instance_variables + num_witness_variables];

        for (i, lagrange_i_tau) in lagrange_vec_tau.iter().enumerate().take(num_constraints) {
            for &(ref a_i_j, j) in &matrices.a[i] {
                u[j] += &(*lagrange_i_tau * a_i_j);
            }
            for &(ref b_i_j, j) in &matrices.b[i] {
                v[j] += &(*lagrange_i_tau * b_i_j);
            }
            for &(ref c_i_j, j) in &matrices.c[i] {
                w[j] += &(*lagrange_i_tau * c_i_j);
            }
        }
        Ok((u, v, w, vanish_tau))
    }
}
