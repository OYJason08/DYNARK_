//! An implementation of the [`Groth16`] zkSNARK.
//!
//! [`Groth16`]: https://eprint.iacr.org/2016/260.pdf
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    missing_docs
)]
#![allow(clippy::many_single_char_names, clippy::op_ref)]
// #![forbid(unsafe_code)]

#[macro_use]
extern crate ark_std;

#[cfg(feature = "r1cs")]
#[macro_use]
extern crate derivative;

/// Reduce an R1CS instance to a *Quadratic Arithmetic Program* instance.
pub mod r1cs_to_qap;

/// Data structures used by the prover, verifier, and generator.
pub mod data_structures;

/// Generate public parameters for the Groth16 zkSNARK construction.
pub mod generator;

/// Create proofs for the Groth16 zkSNARK construction.
pub mod prover;

/// Generate public parameters for the Groth16 zkSNARK construction.
pub mod dynark_generator;
/// Create proofs for the Groth16 zkSNARK construction.
pub mod dynark_prover;
/// Verify proofs for the Groth16 zkSNARK construction.
pub mod dynark_verifier;

/// Verify proofs for the Groth16 zkSNARK construction.
pub mod verifier;

/// FFT_Handler helps prover to handle fft in sequence.
pub mod fft_handler;

///
pub mod utils;

///
pub mod instance_generator;

///Cahce of dynamic keys
pub mod dynamic_cache;

/// Constraints for the Groth16 verifier.
#[cfg(feature = "r1cs")]
pub mod constraints;

#[cfg(test)]
mod test;

pub use self::{data_structures::*, verifier::*};

use ark_crypto_primitives::snark::*;
use ark_ec::pairing::Pairing;
use ark_relations::r1cs::{ConstraintMatrices, ConstraintSynthesizer, SynthesisError};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    vec::Vec,
};
use r1cs_to_qap::{LibsnarkReduction, R1CSToQAP};

/// The SNARK of [[Groth16]](https://eprint.iacr.org/2016/260.pdf).
pub struct Groth16<E: Pairing, QAP: R1CSToQAP = LibsnarkReduction> {
    _p: PhantomData<(E, QAP)>,
}

impl<E: Pairing, QAP: R1CSToQAP> SNARK<E::ScalarField> for Groth16<E, QAP> {
    type ProvingKey = ProvingKey<E>;
    type VerifyingKey = VerifyingKey<E>;
    type Proof = Proof<E>;
    type ProcessedVerifyingKey = PreparedVerifyingKey<E>;
    type Error = SynthesisError;

    fn circuit_specific_setup<C: ConstraintSynthesizer<E::ScalarField>, R: RngCore>(
        circuit: C,
        rng: &mut R,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error> {
        let pk = Self::generate_prover_key(circuit, rng)?;
        let vk = pk.vk.clone();

        Ok((pk, vk))
    }

    fn prove<C: ConstraintSynthesizer<E::ScalarField>, R: RngCore>(
        pk: &Self::ProvingKey,
        circuit: C,
        rng: &mut R,
    ) -> Result<Self::Proof, Self::Error> {
        Self::generate_proof_with_circuit(circuit, pk, rng)
    }

    fn process_vk(
        circuit_vk: &Self::VerifyingKey,
    ) -> Result<Self::ProcessedVerifyingKey, Self::Error> {
        Ok(prepare_verifying_key(circuit_vk))
    }

    fn verify_with_processed_vk(
        circuit_pvk: &Self::ProcessedVerifyingKey,
        x: &[E::ScalarField],
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error> {
        Ok(Self::verify_proof(&circuit_pvk, proof, &x)?)
    }
}

impl<E: Pairing, QAP: R1CSToQAP> CircuitSpecificSetupSNARK<E::ScalarField> for Groth16<E, QAP> {}

/// The basic functionality for a SNARK.
pub trait SemiDynamicSNARK<E: Pairing> {
    /// The information required by the prover to produce a proof for a specific
    /// circuit *C*.
    type ProvingKey: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// The information required by the updater to update a proof for a specific
    /// circuit *C*.
    type UpdatingKey: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// The information required by the verifier to check a proof for a specific
    /// circuit *C*.
    type VerifyingKey: Clone + CanonicalSerialize + CanonicalDeserialize;

    ///
    type Cache: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// The proof output by the prover.
    type Proof: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// This contains the verification key, but preprocessed to enable faster
    /// verification.
    type ProcessedVerifyingKey: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// Errors encountered during setup, proving, or verification.
    type Error: 'static + ark_std::error::Error;

    /// Takes in a description of a computation (specified in R1CS constraints),
    /// and samples proving and verification keys for that circuit.
    fn setup_from_matrices<R: RngCore + CryptoRng>(
        r1cs: ConstraintMatrices<E::ScalarField>,
        rng: &mut R,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey, Self::UpdatingKey), Self::Error>;

    /// Generates a proof of satisfaction of the arithmetic circuit C (specified
    /// as R1CS constraints).
    fn prove<R: RngCore + CryptoRng>(
        r1cs_pk: &Self::ProvingKey,
        r1cs: ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        rng: &mut R,
    ) -> Result<(Self::Proof, Self::Cache), Self::Error>;

    /// Preprocess
    fn process(
        uk: &Self::UpdatingKey,
        r1cs: ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        cache: Self::Cache,
    ) -> Result<Self::Cache, Self::Error>;

    /// Update the proof using the updating key and the cache
    fn update(
        uk: &Self::UpdatingKey,
        r1cs: ConstraintMatrices<E::ScalarField>,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
        cache: Self::Cache,
    ) -> Result<Self::Proof, Self::Error>;

    /// Checks that `proof` is a valid proof of the satisfaction of circuit
    /// encoded in `circuit_vk`, with respect to the public input
    /// `public_input`, specified as R1CS constraints.
    fn verify(
        r1cs_vk: &Self::VerifyingKey,
        instance: &[E::ScalarField],
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error>;
}
