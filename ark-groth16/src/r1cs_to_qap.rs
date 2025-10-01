use ark_ff::PrimeField;
use ark_poly::EvaluationDomain;
use ark_std::{cfg_iter, cfg_iter_mut, vec};

use crate::Vec;
use ark_relations::r1cs::{
    ConstraintMatrices, ConstraintSystemRef, Result as R1CSResult, SynthesisError,
};
use core::ops::Deref;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[inline]
/// compute the inner product of the row of R1CS matrices "terms" with the z
/// "assignment" Computes the inner product of `terms` with `assignment`.
///
/// This implementation is optimized for both parallel and sequential execution:
/// - In parallel mode, it uses Rayon's parallel iterator for efficient
///   multi-threading
/// - In sequential mode, it processes elements in chunks for better
///   vectorization
///
/// # Performance characteristics
/// - Time complexity: O(n) where n is the number of terms
/// - Space complexity: O(1) in sequential mode, O(log n) in parallel mode due
///   to work splitting
///
/// # Arguments
/// * `terms` - Slice of tuples containing coefficients and their indices
/// * `assignment` - Slice of values to be multiplied with coefficients
pub fn evaluate_constraint<F: PrimeField>(terms: &[(F, usize)], assignment: &[F]) -> F {
    #[cfg(feature = "parallel")]
    if terms.len() < 100 {
        serial_evaluate_constraint(terms, assignment)
    } else {
        terms
            .par_iter()
            .map(|(coeff, index)| {
                let val = assignment[*index];
                if coeff.is_one() {
                    val
                } else {
                    val * coeff
                }
            })
            .sum()
    }
    #[cfg(not(feature = "parallel"))]
    serial_evaluate_constraint(terms, assignment)
}

fn serial_evaluate_constraint<F: PrimeField>(terms: &[(F, usize)], assignment: &[F]) -> F {
    let mut sum = F::zero();
    // Process elements in chunks for better CPU vectorization
    for chunk in terms.chunks(4) {
        let chunk_sum = chunk
            .iter()
            .map(|(coeff, index)| {
                let val = assignment[*index];
                if coeff.is_one() {
                    val
                } else {
                    val * coeff
                }
            })
            .sum::<F>();
        sum += chunk_sum;
    }
    sum
}

/// Computes instance and witness reductions from R1CS to
/// Quadratic Arithmetic Programs (QAPs).
pub trait R1CSToQAP {
    ///
    #[inline]
    fn lagrange_form_from_cs<F: PrimeField, D: EvaluationDomain<F>>(
        cs_ref: ConstraintSystemRef<F>,
    ) -> Result<(D, Vec<F>, Vec<F>, Vec<F>), SynthesisError> {
        let matrices = cs_ref.to_matrices().unwrap();
        let num_constraints = cs_ref.num_constraints();

        let cs = cs_ref.borrow().unwrap();
        let prover = cs.deref();

        let instance_witness = [
            prover.instance_assignment.as_slice(),
            prover.witness_assignment.as_slice(),
        ]
        .concat();

        let domain_min_size = num_constraints;
        let domain = D::new(domain_min_size).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_size = domain.size();
        let zero = F::zero();

        // a matrix and z product
        let mut a_lagrange: Vec<F> = vec![zero; domain_size];
        let mut b_lagrange = vec![zero; domain_size];
        let mut c_lagrange = vec![zero; domain_size];

        cfg_iter_mut!(a_lagrange[..num_constraints])
            .zip(cfg_iter!(&matrices.a))
            .for_each(|(a_lagrange_i, a_row)| {
                *a_lagrange_i = evaluate_constraint(&a_row, &instance_witness);
            });

        cfg_iter_mut!(b_lagrange[..num_constraints])
            .zip(cfg_iter!(&matrices.b))
            .for_each(|(b_lagrange_i, b_row)| {
                *b_lagrange_i = evaluate_constraint(&b_row, &instance_witness);
            });

        cfg_iter_mut!(c_lagrange[..num_constraints])
            .zip(cfg_iter!(&matrices.c))
            .for_each(|(c_lagrange_i, c_row)| {
                *c_lagrange_i = evaluate_constraint(&c_row, &instance_witness);
            });

        Ok((domain, a_lagrange, b_lagrange, c_lagrange))
    }

    ///
    #[inline]
    fn compute_quotient<F: PrimeField, D: EvaluationDomain<F>>(
        domain: &D,
        mut a: Vec<F>,
        mut b: Vec<F>,
        mut c: Vec<F>,
    ) -> Result<Vec<F>, SynthesisError> {
        domain.ifft_in_place(&mut a);
        domain.ifft_in_place(&mut b);

        let coset_domain = domain.get_coset(F::GENERATOR).unwrap();

        coset_domain.fft_in_place(&mut a);
        coset_domain.fft_in_place(&mut b);

        let mut ab = domain.mul_polynomials_in_evaluation_domain(&a, &b);
        drop(a);
        drop(b);

        domain.ifft_in_place(&mut c);
        coset_domain.fft_in_place(&mut c);

        let vanishing_polynomial_over_coset = domain
            .evaluate_vanishing_polynomial(F::GENERATOR)
            .inverse()
            .unwrap();
        cfg_iter_mut!(ab).zip(c).for_each(|(ab_i, c_i)| {
            *ab_i -= &c_i;
            *ab_i *= &vanishing_polynomial_over_coset;
        });

        coset_domain.ifft_in_place(&mut ab);

        Ok(ab)
    }

    ///
    #[inline]
    fn preprocessing<F: PrimeField, D: EvaluationDomain<F>>(cs_ref: ConstraintSystemRef<F>) -> () {
        let (domain, a, b, c) = Self::lagrange_form_from_cs::<F, D>(cs_ref).unwrap();
    }

    #[inline]
    /// Computes a QAP witness corresponding to the R1CS witness defined by
    /// `cs`.
    fn compute_quotient_optimized<F: PrimeField, D: EvaluationDomain<F>>(
        prover: ConstraintSystemRef<F>,
    ) -> Result<Vec<F>, SynthesisError> {
        let matrices = prover.to_matrices().unwrap();
        let num_constraints = prover.num_constraints();

        let cs = prover.borrow().unwrap();
        let prover = cs.deref();

        let instance_witness = [
            prover.instance_assignment.as_slice(),
            prover.witness_assignment.as_slice(),
        ]
        .concat();

        let domain_min_size = num_constraints;
        let domain = D::new(domain_min_size).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_size = domain.size();
        let zero = F::zero();

        // a matrix and z product
        let mut a = vec![zero; domain_size];
        let mut b = vec![zero; domain_size];

        cfg_iter_mut!(a[..num_constraints])
            .zip(cfg_iter_mut!(b[..num_constraints]))
            .zip(cfg_iter!(&matrices.a))
            .zip(cfg_iter!(&matrices.b))
            .for_each(|(((a, b), a_row), b_row)| {
                *a = evaluate_constraint(&a_row, &instance_witness);
                *b = evaluate_constraint(&b_row, &instance_witness);
            });

        domain.ifft_in_place(&mut a);
        domain.ifft_in_place(&mut b);

        let coset_domain = domain.get_coset(F::GENERATOR).unwrap();

        coset_domain.fft_in_place(&mut a);
        coset_domain.fft_in_place(&mut b);

        let mut ab = domain.mul_polynomials_in_evaluation_domain(&a, &b);
        drop(a);
        drop(b);

        let mut c = vec![zero; domain_size];
        cfg_iter_mut!(c[..num_constraints])
            .enumerate()
            .for_each(|(i, c)| {
                *c = evaluate_constraint(&matrices.c[i], &instance_witness);
            });

        domain.ifft_in_place(&mut c);
        coset_domain.fft_in_place(&mut c);

        let vanishing_polynomial_over_coset = domain
            .evaluate_vanishing_polynomial(F::GENERATOR)
            .inverse()
            .unwrap();
        cfg_iter_mut!(ab).zip(c).for_each(|(ab_i, c_i)| {
            *ab_i -= &c_i;
            *ab_i *= &vanishing_polynomial_over_coset;
        });

        coset_domain.ifft_in_place(&mut ab);

        Ok(ab)
    }

    #[inline]
    /// Computes a QAP witness corresponding to the R1CS witness defined by
    /// `cs`.
    fn compute_quotient_arkwork<F: PrimeField, D: EvaluationDomain<F>>(
        prover: ConstraintSystemRef<F>,
    ) -> Result<Vec<F>, SynthesisError> {
        let matrices = prover.to_matrices().unwrap();
        let num_instance_variables = prover.num_instance_variables();
        let num_constraints = prover.num_constraints();

        let cs = prover.borrow().unwrap();
        let prover = cs.deref();

        let instance_witness = [
            prover.instance_assignment.as_slice(),
            prover.witness_assignment.as_slice(),
        ]
        .concat();

        Self::compute_quotient_from_matrices::<F, D>(
            &matrices,
            num_instance_variables,
            num_constraints,
            &instance_witness,
        )
    }

    /// Computes a QAP witness corresponding to the R1CS witness defined by
    /// `cs`.
    fn compute_quotient_from_matrices<F: PrimeField, D: EvaluationDomain<F>>(
        matrices: &ConstraintMatrices<F>,
        num_instance_variables: usize,
        num_constraints: usize,
        instance_witness: &[F],
    ) -> R1CSResult<Vec<F>>;
}

/// Computes the exponents that the generator uses to calculate base
/// elements which the prover later uses to compute `h(x)t(x)/delta`.
pub fn tau_powers_field_lifted<F: PrimeField, D: EvaluationDomain<F>>(
    max_power: usize,
    tau: F,
    vanish_tau: F,
    delta_inverse: F,
) -> Result<Vec<F>, SynthesisError> {
    let scalars = cfg_into_iter!(0..max_power)
        .map(|i| vanish_tau * &delta_inverse * &tau.pow([i as u64]))
        .collect::<Vec<_>>();
    Ok(scalars)
}

///
pub fn tau_powers_field<F: PrimeField, D: EvaluationDomain<F>>(
    max_power: usize,
    tau: F,
    delta_inverse: F,
) -> Result<Vec<F>, SynthesisError> {
    let scalars = cfg_into_iter!(0..max_power)
        .map(|i| delta_inverse * &tau.pow([i as u64]))
        .collect::<Vec<_>>();
    Ok(scalars)
}

/// Computes the R1CS-to-QAP reduction defined in [`libsnark`](https://github.com/scipr-lab/libsnark/blob/2af440246fa2c3d0b1b0a425fb6abd8cc8b9c54d/libsnark/reductions/r1cs_to_qap/r1cs_to_qap.tcc).
pub struct LibsnarkReduction;

impl R1CSToQAP for LibsnarkReduction {
    // #[inline]
    // #[allow(clippy::type_complexity)]
    // fn compute_u_v_w<F: PrimeField, D: EvaluationDomain<F>>(
    //     cs: ConstraintSystemRef<F>,
    //     tau: &F,
    // ) -> R1CSResult<(Vec<F>, Vec<F>, Vec<F>, F, usize, usize)> {
    //     // Get info
    //     let num_constraints = cs.num_constraints();
    //     let num_instance_variables = cs.num_instance_variables();
    //     let num_witness_variables = cs.num_witness_variables();
    //     let qap_num_variables = (num_instance_variables - 1) +
    // num_witness_variables; // the fixed value 1 is not counted for
    // qap_num_variables     let matrices = cs.to_matrices().unwrap(); // the
    // row_num should be num_constraints; the col_num should be
    // // qap_num_variables + 1

    //     debug_assert!(matrices.num_constraints == num_constraints);
    //     debug_assert!(matrices.num_instance_variables == num_instance_variables);
    //     debug_assert!(matrices.num_witness_variables == num_witness_variables);

    //     // Generate domain

    //     // @@@@@@@@@@@@@@@@@@@@@@@
    //     // let domain_min_size = num_constraints + num_instance_variables;
    //     let domain_min_size = num_constraints;
    //     let domain =
    // D::new(domain_min_size).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    //     let domain_size = domain.size(); // domain_size >= num_constraints +
    // num_instance_variables

    //     let vanish_tau = domain.evaluate_vanishing_polynomial(*tau);

    //     // Evaluate all Lagrange polynomials
    //     let coefficients_time = start_timer!(|| "Evaluate Lagrange
    // coefficients");     let lagrange_vec_tau: Vec<F> =
    // domain.evaluate_all_lagrange_coefficients(*tau);     end_timer!
    // (coefficients_time);

    //     let mut u: Vec<F> = vec![F::zero(); qap_num_variables + 1];
    //     let mut v: Vec<F> = vec![F::zero(); qap_num_variables + 1];
    //     let mut w: Vec<F> = vec![F::zero(); qap_num_variables + 1];

    //     // @@@@@@@@@@@@@@@@@@@@@@@

    //     // u[0..num_instance_variables].copy_from_slice(
    //     //     &lagrange_vec_tau[num_constraints..(num_instance_variables +
    //     // num_constraints)], );

    //     for (i, lagrange_i_tau) in
    // lagrange_vec_tau.iter().enumerate().take(num_constraints) {         for
    // &(ref a_i_j, j) in &matrices.a[i] {             u[j] += &(*lagrange_i_tau
    // * a_i_j);         } for &(ref b_i_j, j) in &matrices.b[i] { v[j] +=
    //   &(*lagrange_i_tau * b_i_j); } for &(ref c_i_j, j) in &matrices.c[i] { w[j]
    //   += &(*lagrange_i_tau * c_i_j); } }

    //     Ok((u, v, w, vanish_tau, qap_num_variables, domain_size))
    // }

    fn compute_quotient_from_matrices<F: PrimeField, D: EvaluationDomain<F>>(
        matrices: &ConstraintMatrices<F>,
        num_instance_variables: usize,
        num_constraints: usize,
        instance_witness: &[F],
    ) -> R1CSResult<Vec<F>> {
        // @@@@@@@@@@@@@@@@@@@@@@@
        // let domain_min_size = num_constraints + num_instance_variables;
        let domain_min_size = num_constraints;
        let domain = D::new(domain_min_size).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_size = domain.size();
        let zero = F::zero();

        // a matrix and z product
        let mut a = vec![zero; domain_size];
        let mut b = vec![zero; domain_size];

        cfg_iter_mut!(a[..num_constraints])
            .zip(cfg_iter_mut!(b[..num_constraints]))
            .zip(cfg_iter!(&matrices.a))
            .zip(cfg_iter!(&matrices.b))
            .for_each(|(((a, b), a_row), b_row)| {
                *a = evaluate_constraint(&a_row, &instance_witness);
                *b = evaluate_constraint(&b_row, &instance_witness);
            });

        // @@@@@@@@@@@@@@@@@@@@@@@
        // a[num_constraints..num_constraints + num_instance_variables]
        //     .clone_from_slice(&instance_witness[..num_instance_variables]);

        domain.ifft_in_place(&mut a);
        domain.ifft_in_place(&mut b);

        let coset_domain = domain.get_coset(F::GENERATOR).unwrap();

        coset_domain.fft_in_place(&mut a);
        coset_domain.fft_in_place(&mut b);

        let mut ab = domain.mul_polynomials_in_evaluation_domain(&a, &b);
        drop(a);
        drop(b);

        let mut c = vec![zero; domain_size];
        cfg_iter_mut!(c[..num_constraints])
            .enumerate()
            .for_each(|(i, c)| {
                *c = evaluate_constraint(&matrices.c[i], &instance_witness);
            });

        domain.ifft_in_place(&mut c);
        coset_domain.fft_in_place(&mut c);

        let vanishing_polynomial_over_coset = domain
            .evaluate_vanishing_polynomial(F::GENERATOR)
            .inverse()
            .unwrap();
        cfg_iter_mut!(ab).zip(c).for_each(|(ab_i, c_i)| {
            *ab_i -= &c_i;
            *ab_i *= &vanishing_polynomial_over_coset;
        });

        coset_domain.ifft_in_place(&mut ab);

        Ok(ab)
    }

    // fn tau_powers_field<F: PrimeField, D: EvaluationDomain<F>>(
    //     max_power: usize,
    //     t: F,
    //     zt: F,
    //     delta_inverse: F,
    // ) -> Result<Vec<F>, SynthesisError> {
    //     let scalars = cfg_into_iter!(0..max_power)
    //         .map(|i| zt * &delta_inverse * &t.pow([i as u64]))
    //         .collect::<Vec<_>>();
    //     Ok(scalars)
    // }
}
