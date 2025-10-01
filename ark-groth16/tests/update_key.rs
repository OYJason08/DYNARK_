
use ark_bls12_381::Bls12_381;
use ark_poly::{
    EvaluationDomain, GeneralEvaluationDomain,
};
use ark_std::rand::Rng;

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{Field, UniformRand, Zero};
use ark_groth16::{
    r1cs_to_qap::{tau_powers_field, tau_powers_field_lifted},
    Groth16,
};
use ark_relations::r1cs::{
    ConstraintMatrices, Matrix, SynthesisError,
};
use ark_std::test_rng;

fn test_quotient_decompostion() {
    type E = Bls12_381;
    type F = <E as Pairing>::ScalarField;

    let instance = vec![F::from(1u64), F::from(3u64)];
    let witness = vec![F::from(4u64)];
    let z = vec![instance[0], instance[1], witness[0]];

    let a: Matrix<F> = vec![
        vec![(F::from(2u64), 0), (F::from(1u64), 1)], // row 1
        vec![(F::from(1u64), 1)],                     // row 2
        vec![(F::from(7u64), 0)],                     // row 3
        vec![(F::from(1u64), 2)],                     // row 4
    ];
    let b: Matrix<F> = vec![
        vec![(F::from(3u64), 2)],                                         // row 1
        vec![(F::from(5u64), 0), (F::from(1u64), 2)],                     // row 2
        vec![(F::from(2u64), 0), (F::from(1u64), 1), (F::from(1u64), 2)], // row 3
        vec![(F::from(2u64), 1)],                                         // row 4
    ];
    let c: Matrix<F> = vec![
        vec![(F::from(60u64), 0)], // row 1
        vec![(F::from(27u64), 0)], // row 2
        vec![(F::from(63u64), 0)], // row 3
        vec![(F::from(24u64), 0)], // row 4
    ];

    let matrices = ConstraintMatrices {
        num_instance_variables: instance.len(),
        num_witness_variables: witness.len(),
        num_constraints: 4,
        a_num_non_zero: a.iter().map(|row| row.len()).sum(),
        b_num_non_zero: b.iter().map(|row| row.len()).sum(),
        c_num_non_zero: c.iter().map(|row| row.len()).sum(),
        a,
        b,
        c,
    };

    let inner =
        |row: &Vec<(F, usize)>| -> F { row.iter().map(|(coeff, idx)| *coeff * z[*idx]).sum() };

    for i in 0..matrices.num_constraints {
        let ai = inner(&matrices.a[i]);
        let bi = inner(&matrices.b[i]);
        let ci = inner(&matrices.c[i]);
        println!("Row {}: A z = {}, B z = {}, C z = {}", i + 1, ai, bi, ci);
        assert_eq!(ai * bi, ci);
    }

    let mut rng = test_rng();

    quotient_decomposition::<E>(instance, witness, &matrices, &mut rng);
}

fn quotient_decomposition<E>(
    instance: Vec<E::ScalarField>,
    witness: Vec<E::ScalarField>,
    matrices: &ConstraintMatrices<E::ScalarField>,
    rng: &mut impl Rng,
) where
    E: Pairing,
{
    let num_constraints = matrices.num_constraints;
    let num_instance_variables = matrices.num_instance_variables;
    let num_witness_variables = matrices.num_witness_variables;

    let domain_min_size = matrices.num_constraints;
    let domain = GeneralEvaluationDomain::<E::ScalarField>::new(domain_min_size)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    let domain_size = domain.size();

    let tau = domain.sample_element_outside_domain(rng);
    let alpha = E::ScalarField::rand(rng);
    let beta = E::ScalarField::rand(rng);
    let gamma = E::ScalarField::rand(rng);
    let delta = E::ScalarField::rand(rng);

    let g1_gen = E::G1::rand(rng).into_affine();
    let g2_gen = E::G2::rand(rng).into_affine();

    let prover_key = Groth16::<E>::generate_prover_key_with_matrices(
        domain,
        &matrices,
        (tau, alpha, beta, gamma, delta, g1_gen, g2_gen),
    )
    .unwrap();

    let delta_inverse = delta
        .inverse()
        .ok_or(SynthesisError::UnexpectedIdentity)
        .unwrap();

    let updating_key = Groth16::<E>::generate_updating_keys(&matrices, &domain, &prover_key).unwrap();

    let vanish_tau = domain.evaluate_vanishing_polynomial(tau);

    let mut tau_powers_lifted_truncated = tau_powers_field_lifted::<
        _,
        GeneralEvaluationDomain<E::ScalarField>,
    >(domain_size - 1, tau, vanish_tau, delta_inverse)
    .unwrap();
    tau_powers_lifted_truncated.push(E::ScalarField::zero());

    let tau_powers_complete = tau_powers_field::<_, GeneralEvaluationDomain<E::ScalarField>>(
        domain_size,
        tau,
        delta_inverse,
    )
    .unwrap();

    let w = updating_key.w.clone();

    let lagrange_complete = domain.ifft(&tau_powers_complete);
    let lagrange_truncated = domain.ifft(&tau_powers_lifted_truncated);

    for i in 0..domain_size {
        for j in 0..domain_size {
            if i != j {
                let lagrange_product = lagrange_complete[i] * lagrange_complete[j];
                let j_i = if j >= i { j - i } else { j + domain_size - i };
                let i_j = if i >= j { i - j } else { i + domain_size - j };
                let decomposition = E::ScalarField::from(domain_size as u64).inverse().unwrap()
                    * (w[j_i] * lagrange_truncated[i] + w[i_j] * lagrange_truncated[j]);
                println!(
                    "i = {}, j = {}, lagrange_product = {}, decomposition = {}",
                    i, j, lagrange_product, decomposition
                );
                assert_eq!(lagrange_product, decomposition);
            }
        }
    }
}
