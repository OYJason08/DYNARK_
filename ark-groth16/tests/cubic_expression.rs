use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_ec::pairing::Pairing;
use ark_ff::{Field, One}; // Import One trait
use ark_groth16::{prepare_verifying_key, Groth16};
use ark_relations::{
    lc,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable}, /* Import Variable */
};
use ark_serialize::CanonicalSerialize;
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng, UniformRand,
};
use std::time::Instant;

/// Define a simple circuit that computes x^3 + x + 1 = y
struct CubicPlusLinearCircuit<F: Field> {
    x: Option<F>,
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF>
    for CubicPlusLinearCircuit<ConstraintF>
{
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        // Define variables x, x^2, x^3 and y (i.e. x^3 + x + 1)
        let x = cs.new_witness_variable(|| self.x.ok_or(SynthesisError::AssignmentMissing))?;
        let x_squared = cs.new_witness_variable(|| {
            let mut x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            x_val.square_in_place(); // Calculate x^2
            Ok(x_val)
        })?;
        let x_cubed = cs.new_witness_variable(|| {
            let mut x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            x_val.square_in_place(); // Calculate x^2
            x_val *= &self.x.ok_or(SynthesisError::AssignmentMissing)?; // Calculate x^3
            Ok(x_val)
        })?;
        let y = cs.new_input_variable(|| {
            let mut x_cubed_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            x_cubed_val.square_in_place(); // Calculate x^2
            x_cubed_val *= &self.x.ok_or(SynthesisError::AssignmentMissing)?; // Calculate x^3

            let mut result = x_cubed_val; // Initial value is x^3
            result += &self.x.ok_or(SynthesisError::AssignmentMissing)?; // Add x
            result += ConstraintF::one(); // Add 1
            Ok(result) // Return y
        })?;

        // Add constraints: x * x = x^2, x^2 * x = x^3, and x^3 + x + 1 = y
        cs.enforce_constraint(lc!() + x, lc!() + x, lc!() + x_squared)?; // x * x = x^2
        cs.enforce_constraint(lc!() + x_squared, lc!() + x, lc!() + x_cubed)?; // x^2 * x = x^3
        cs.enforce_constraint(
            lc!() + x_cubed + x + (ConstraintF::one(), Variable::One), // x^3 + x + 1
            lc!() + Variable::One,                                     /* 1 (no multiplication
                                                                        * needed) */
            lc!() + y,
        )?; // y = x^3 + x + 1

        // Print number of constraints
        println!("Number of constraints: {}", cs.num_constraints());
        Ok(())
    }
}

/// test
#[test]
pub fn test_cubic_expression() {
    // Use BLS12-381 elliptic curve
    test_prove_and_verify::<ark_bls12_381::Bls12_381>();
}

/// Proof and verification function
fn test_prove_and_verify<E>()
where
    E: Pairing,
{
    // Create random number generator
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    // Execute Groth16 setup (generate proving key and verifying key)
    let (pk, vk) = Groth16::<E>::setup(CubicPlusLinearCircuit { x: None }, &mut rng).unwrap();

    let pvk = prepare_verifying_key::<E>(&vk);

    // Calculate and print uncompressed sizes of pk and vk
    let pk_size = pk.uncompressed_size();
    let vk_size = vk.uncompressed_size();
    println!("Uncompressed pk size: {} bytes", pk_size);
    println!("Uncompressed vk size: {} bytes", vk_size);
    println!(
        "Total uncompressed size (pk + vk): {} bytes",
        pk_size + vk_size
    );

    // Generate random x
    let x = E::ScalarField::rand(&mut rng);
    let mut y = x;
    y.square_in_place(); // Calculate x^2
    let x_cubed = y * x; // Calculate x^3
    y = x_cubed + x + E::ScalarField::one(); // Calculate y = x^3 + x + 1

    let start1 = Instant::now();
    // Use Groth16 to prove x^3 + x + 1 = y
    let proof = Groth16::<E>::prove(&pk, CubicPlusLinearCircuit { x: Some(x) }, &mut rng).unwrap();
    let start2 = Instant::now();
    // Output uncompressed proof size
    let uncompressed_size = proof.uncompressed_size();
    println!("Uncompressed proof size: {} bytes", uncompressed_size);
    // Verify the proof
    let is_valid = Groth16::<E>::verify_with_processed_vk(&pvk, &[y], &proof).unwrap();
    let start3 = Instant::now();
    println!("Proof is valid: {}", is_valid); // Print proof result
    let duration1 = start2.duration_since(start1);
    let duration2 = start3.duration_since(start2);

    let millis1 = duration1.as_secs_f64() * 1000.0;
    let millis2 = duration2.as_secs_f64() * 1000.0;
    println!("Prove time: {:.3} milliseconds", millis1);
    println!("Verify time: {:.3} milliseconds", millis2);
}
