use crate::{
    r1cs_to_qap::R1CSToQAP,
    Groth16, ProvingKey,
};
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::UniformRand;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{
    ConstraintMatrices, ConstraintSynthesizer, ConstraintSystem, OptimizationGoal,
    Result as R1CSResult, SynthesisError, SynthesisMode,
};
use ark_std::rand::Rng;

impl<E: Pairing, QAP: R1CSToQAP> Groth16<E, QAP> {
    ///
    pub fn process_circuit<C>(
        circuit: C,
    ) -> (
        GeneralEvaluationDomain<E::ScalarField>,
        ConstraintMatrices<E::ScalarField>,
    )
    where
        C: ConstraintSynthesizer<E::ScalarField>,
    {
        let process_circuit_time = start_timer!(|| "Groth16::process_circuit");

        let cs = ConstraintSystem::new_ref();
        cs.set_optimization_goal(OptimizationGoal::Constraints);
        cs.set_mode(SynthesisMode::Setup);
        circuit.generate_constraints(cs.clone()).unwrap();
        cs.finalize();

        let domain_min_size = cs.num_constraints();
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(domain_min_size)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();

        let matrices = cs.to_matrices().unwrap(); // the row_num should be num_constraints; the col_num should be

        end_timer!(process_circuit_time);

        (domain, matrices)
    }

    /// Generates a random common reference string for
    /// a circuit using the provided R1CS-to-QAP reduction.
    #[inline]
    pub fn generate_prover_key<C>(circuit: C, rng: &mut impl Rng) -> R1CSResult<ProvingKey<E>>
    where
        C: ConstraintSynthesizer<E::ScalarField>,
    {
        let (domain, matrices) = Self::process_circuit(circuit);

        let tau = domain.sample_element_outside_domain(rng);
        let alpha = E::ScalarField::rand(rng);
        let beta = E::ScalarField::rand(rng);
        let gamma = E::ScalarField::rand(rng);
        let delta = E::ScalarField::rand(rng);

        let g1_gen = E::G1::rand(rng).into_affine();
        let g2_gen = E::G2::rand(rng).into_affine();

        Self::generate_prover_key_with_matrices(
            domain,
            &matrices,
            (tau, alpha, beta, gamma, delta, g1_gen, g2_gen),
        )
    }
}
