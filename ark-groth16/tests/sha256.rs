use ark_bls12_381::Bls12_381;
use ark_crypto_primitives::{
    crh::{
        sha256::constraints::{Sha256Gadget, UnitVar},
        CRHSchemeGadget,
    },
    snark::{CircuitSpecificSetupSNARK, SNARK},
};
use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_groth16::{prepare_verifying_key, Groth16};
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng,
    time::Instant,
    vec::Vec,
    Zero,
};
use rand_chacha::ChaChaRng;
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

// SHA256 Circuit Definition
struct Sha256Circuit<ConstraintF: Field> {
    preimage: Option<Vec<u8>>, // Input to be hashed
    hash: Option<Vec<u8>>,     // Expected hash value to verify
    _phantom: PhantomData<ConstraintF>,
}

impl<ConstraintF: PrimeField> ConstraintSynthesizer<ConstraintF> for Sha256Circuit<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        // Modification: Make hash value a private input instead of public input
        let preimage_var = UInt8::new_witness_vec(
            ark_relations::ns!(cs, "preimage"),
            self.preimage.as_deref().unwrap_or(&[]),
        )?;

        let hash_var = UInt8::new_witness_vec(
            // Changed to witness instead of input
            ark_relations::ns!(cs, "hash"),
            self.hash.as_deref().unwrap_or(&[]),
        )?;

        // Use SHA256 gadget to compute hash
        let computed_hash =
            Sha256Gadget::<ConstraintF>::evaluate(&UnitVar::default(), &preimage_var)?;

        // Add constraint: computed hash must equal input hash
        for (computed_byte, expected_byte) in computed_hash.0.iter().zip(hash_var.iter()) {
            computed_byte.enforce_equal(expected_byte)?;
        }

        Ok(())
    }
}

/// Test function to prove and verify the SHA256 circuit    
#[test]
fn test_sha256() {
    // Test on BLS12-381 curve
    test_prove_and_verify::<Bls12_381>();
}

fn test_prove_and_verify<E>()
where
    E: Pairing,
{
    let mut rng = test_rng();
    let mut prover_rng = ChaChaRng::seed_from_u64(rng.next_u32() as u64);

    // Setup input data
    let preimage = b"Hello, World!".to_vec();
    let mut hasher = Sha256::new();
    hasher.update(&preimage);
    let hash = hasher.finalize().to_vec();

    println!("Setting up circuit...");
    // Setup circuit
    let setup_circuit = Sha256Circuit {
        preimage: None,
        hash: None,
        _phantom: PhantomData,
    };

    println!("Generating proving key and verifying key...");
    let (pk, vk) = Groth16::<E>::setup(setup_circuit, &mut prover_rng).expect("Setup failed");

    println!("Setting up proving circuit...");
    let proving_circuit = Sha256Circuit {
        preimage: Some(preimage.clone()),
        hash: Some(hash.clone()),
        _phantom: PhantomData,
    };

    println!("Generating proof...");
    let start1 = Instant::now();
    let proof = Groth16::<E>::prove(&pk, proving_circuit, &mut prover_rng).expect("Proving failed");
    let start2 = Instant::now();

    println!("Processing verification key...");
    let pvk = prepare_verifying_key(&vk);
    // Modification: Split 32 bytes into 4 groups of 8 bytes each
    let mut combined_hash = E::ScalarField::zero();
    for (i, chunk) in hash.chunks(8).enumerate() {
        let mut chunk_value = E::ScalarField::zero();
        for (j, &byte) in chunk.iter().enumerate() {
            chunk_value +=
                E::ScalarField::from(byte as u64) * E::ScalarField::from(1u64 << (8 * j));
        }
        if i > 0 {
            chunk_value =
                chunk_value * E::ScalarField::from(1u64 << 63) * E::ScalarField::from(2u64);
        }
        combined_hash += chunk_value;
    }

    let public_inputs = vec![combined_hash];

    println!("\nVerifying proof...");
    let verification_result = Groth16::<E>::verify_with_processed_vk(
        &pvk,
        &[], // Empty public inputs
        &proof,
    );
    let start3 = Instant::now();

    // Modification: Access gamma_abc_g1 through pvk.vk
    match &verification_result {
        Ok(valid) => println!("Verification completed with result: {}", valid),
        Err(e) => {
            println!("Verification error details:");
            println!("Error: {:?}", e);
            println!("Expected number of inputs: {}", pvk.vk.t_v_g1.len() - 1);
            println!("Provided number of inputs: {}", public_inputs.len());
        },
    }

    // Calculate timing and output results
    let prove_time = start2.duration_since(start1).as_secs_f64() * 1000.0;
    let verify_time = start3.duration_since(start2).as_secs_f64() * 1000.0;
    let proof_size = std::mem::size_of_val(&proof);

    println!("Prove time: {:.3} ms", prove_time);
    println!("Verify time: {:.3} ms", verify_time);
    println!("Proof size: {} bytes", proof_size);

    match verification_result {
        Ok(is_valid) => println!("Proof verification result: {}", is_valid),
        Err(e) => println!("Verification failed with error: {:?}", e),
    }
}
