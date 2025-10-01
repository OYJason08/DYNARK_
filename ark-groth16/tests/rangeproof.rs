use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_groth16::{prepare_verifying_key, Groth16};
use ark_relations::{
    lc,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable},
};
use ark_serialize::CanonicalSerialize;
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng,
};
use std::time::Instant;

/// Define a circuit to prove x is in range [0, 2^32]
struct RangeProofCircuit<F: Field> {
    x: Option<F>,
}

impl<ConstraintF: PrimeField> ConstraintSynthesizer<ConstraintF>
    for RangeProofCircuit<ConstraintF>
{
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        // Declare input variable x
        let x = cs.new_input_variable(|| self.x.ok_or(SynthesisError::AssignmentMissing))?;

        // Decompose x into 32 binary bits
        let mut bits = Vec::new();
        for i in 0..32 {
            let bit = cs.new_witness_variable(|| {
                let x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
                let x_bigint = x_val.into_bigint();
                let bit_val = (x_bigint.as_ref()[0] >> i) & 1 == 1;
                Ok(if bit_val {
                    ConstraintF::one()
                } else {
                    ConstraintF::zero()
                })
            })?;

            // Constrain each bit to be 0 or 1: bit * (1 - bit) = 0
            cs.enforce_constraint(
                lc!() + bit,
                lc!() + (ConstraintF::one(), Variable::One) - bit,
                lc!(),
            )?;

            bits.push(bit);
        }

        // Constrain the combination of bits equals x
        let mut lc = lc!();
        let mut coeff = ConstraintF::one();
        for bit in bits.iter() {
            lc = lc + (coeff, *bit);
            coeff = coeff.double();
        }

        // Ensure the combination of bits equals x
        cs.enforce_constraint(
            lc!() + lc,
            lc!() + (ConstraintF::one(), Variable::One),
            lc!() + x,
        )?;

        // Print number of constraints
        println!("Number of constraints: {}", cs.num_constraints());
        Ok(())
    }
}

/// Test function to prove and verify the range proof
#[test]
fn test_range_proof() {
    test_prove_and_verify::<ark_bls12_381::Bls12_381>();
}

/// Proof and verification function
fn test_prove_and_verify<E>()
where
    E: Pairing,
{
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    // Setup circuit
    let (pk, vk) = Groth16::<E>::setup(RangeProofCircuit { x: None }, &mut rng).unwrap();

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

    // Generate a random number in range [0, 2^32]
    let x = E::ScalarField::from(rng.next_u32() as u64);

    let start1 = Instant::now();
    // Generate proof
    let proof = Groth16::<E>::prove(&pk, RangeProofCircuit { x: Some(x) }, &mut rng).unwrap();
    let start2 = Instant::now();
    // Output uncompressed proof size
    let uncompressed_size = proof.uncompressed_size();
    println!("Uncompressed proof size: {} bytes", uncompressed_size);
    // Verify the proof
    let is_valid = Groth16::<E>::verify_with_processed_vk(&pvk, &[x], &proof).unwrap();
    let start3 = Instant::now();
    println!("Proof is valid: {}", is_valid); // Print proof result
    let duration1 = start2.duration_since(start1);
    let duration2 = start3.duration_since(start2);

    let millis1 = duration1.as_secs_f64() * 1000.0;
    let millis2 = duration2.as_secs_f64() * 1000.0;
    println!("Prove time: {:.3} milliseconds", millis1);
    println!("Verify time: {:.3} milliseconds", millis2);
}
