
use ark_crypto_primitives::sponge::Absorb;
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::*;
use ark_std::vec::Vec;
use rayon::prelude::*;
/// A proof in the Groth16 SNARK.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof<E: Pairing> {
    /// The `A` element in `G1`.
    pub a_g1: E::G1Affine,
    /// The `B` element in `G2`.
    pub b_g2: E::G2Affine,
    /// The `C` element in `G1`.
    pub c_g1: E::G1Affine,
}

impl<E: Pairing> Default for Proof<E> {
    fn default() -> Self {
        Self {
            a_g1: E::G1Affine::default(),
            b_g2: E::G2Affine::default(),
            c_g1: E::G1Affine::default(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// A verification key in the Groth16 SNARK.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct VerifyingKey<E: Pairing> {
    ///
    pub num_constraints: usize,
    ///
    pub num_instance: usize,
    ///
    pub num_witness: usize,
    ///
    pub domain: GeneralEvaluationDomain<E::ScalarField>,
    /// The `alpha * G`, where `G` is the generator of `E::G1`.
    pub alpha_g1: E::G1Affine,
    /// The `alpha * H`, where `H` is the generator of `E::G2`.
    pub beta_g2: E::G2Affine,
    /// The `gamma * H`, where `H` is the generator of `E::G2`.
    pub gamma_g2: E::G2Affine,
    /// The `delta * H`, where `H` is the generator of `E::G2`.
    pub delta_g2: E::G2Affine,
    /// The `gamma^{-1} * (beta * a_i + alpha * b_i + c_i) * H`, where `H` is
    /// the generator of `E::G1`.
    pub t_v_g1: Vec<E::G1Affine>,
}

impl<E: Pairing> Default for VerifyingKey<E> {
    fn default() -> Self {
        Self {
            num_constraints: 0,
            num_instance: 0,
            num_witness: 0,
            domain: GeneralEvaluationDomain::new(1).unwrap(),
            alpha_g1: E::G1Affine::default(),
            beta_g2: E::G2Affine::default(),
            gamma_g2: E::G2Affine::default(),
            delta_g2: E::G2Affine::default(),
            t_v_g1: Vec::new(),
        }
    }
}

impl<E> Absorb for VerifyingKey<E>
where
    E: Pairing,
    E::G1Affine: Absorb,
    E::G2Affine: Absorb,
{
    fn to_sponge_bytes(&self, dest: &mut Vec<u8>) {
        self.alpha_g1.to_sponge_bytes(dest);
        self.beta_g2.to_sponge_bytes(dest);
        self.gamma_g2.to_sponge_bytes(dest);
        self.delta_g2.to_sponge_bytes(dest);
        self.t_v_g1.iter().for_each(|g| g.to_sponge_bytes(dest));
    }

    fn to_sponge_field_elements<F: PrimeField>(&self, dest: &mut Vec<F>) {
        self.alpha_g1.to_sponge_field_elements(dest);
        self.beta_g2.to_sponge_field_elements(dest);
        self.gamma_g2.to_sponge_field_elements(dest);
        self.delta_g2.to_sponge_field_elements(dest);
        self.t_v_g1
            .iter()
            .for_each(|g| g.to_sponge_field_elements(dest));
    }
}

/// Preprocessed verification key parameters that enable faster verification
/// at the expense of larger size in memory.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct PreparedVerifyingKey<E: Pairing> {
    /// The unprepared verification key.
    pub vk: VerifyingKey<E>,
    /// The element `e(alpha * G, beta * H)` in `E::GT`.
    pub alpha_g1_beta_g2: E::TargetField,
    /// The element `- gamma * H` in `E::G2`, prepared for use in pairings.
    pub gamma_g2_neg_pc: E::G2Prepared,
    /// The element `- delta * H` in `E::G2`, prepared for use in pairings.
    pub delta_g2_neg_pc: E::G2Prepared,
}

impl<E: Pairing> From<PreparedVerifyingKey<E>> for VerifyingKey<E> {
    fn from(other: PreparedVerifyingKey<E>) -> Self {
        other.vk
    }
}

impl<E: Pairing> From<VerifyingKey<E>> for PreparedVerifyingKey<E> {
    fn from(other: VerifyingKey<E>) -> Self {
        crate::prepare_verifying_key(&other)
    }
}

impl<E: Pairing> Default for PreparedVerifyingKey<E> {
    fn default() -> Self {
        Self {
            vk: VerifyingKey::default(),
            alpha_g1_beta_g2: E::TargetField::default(),
            gamma_g2_neg_pc: E::G2Prepared::default(),
            delta_g2_neg_pc: E::G2Prepared::default(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// The prover key for for the Groth16 zkSNARK.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProvingKey<E: Pairing> {
    /// The underlying verification key.
    pub vk: VerifyingKey<E>,
    /// The element `beta * G` in `E::G1`.
    pub beta_g1: E::G1Affine,
    /// The element `delta * G` in `E::G1`.
    pub delta_g1: E::G1Affine,
    /// The elements `a_i * G` in `E::G1`.
    pub u_g1: Vec<E::G1Affine>,
    /// The elements `b_i * G` in `E::G1`.
    pub v_g1: Vec<E::G1Affine>,
    /// The elements `b_i * H` in `E::G2`.
    pub v_g2: Vec<E::G2Affine>,
    /// The elements `h_i * G` in `E::G1`.
    pub tau_powers_g1: Vec<E::G1Affine>,
    /// The elements `l_i * G` in `E::G1`.
    pub t_p_g1: Vec<E::G1Affine>,
}

////////////////////////////////////////////////////////////////////////////////

/// The updater key for for the Groth16 zkSNARK.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct UpdatingKey<E: Pairing> {
    ///
    pub pk: ProvingKey<E>,
    ///
    pub lagrange_truncated: Vec<E::G1>,
    ///
    pub w: Vec<E::ScalarField>,
    ///
    pub w_reverse: Vec<E::ScalarField>,
    ///
    pub p: Vec<E::G1>,
    ///
    pub w_for_q: Vec<E::ScalarField>,
    ///
    pub w_reverse_for_q: Vec<E::ScalarField>,
    ///
    pub a_matrix_cols: Vec<Vec<(usize,E::ScalarField)>>,
    ///
    pub b_matrix_cols: Vec<Vec<(usize,E::ScalarField)>>,
}

///
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Cache<E: Pairing> {
    /// domain
    pub domain: GeneralEvaluationDomain<E::ScalarField>,
    /// number of instance variables
    pub num_instance: usize,
    /// the randomness used to add zero-knowledge
    pub r: E::ScalarField,
    /// the randomness used to add zero-knowledge
    pub s: E::ScalarField,
    /// projection quotients of a_lagrange
    pub q_a: Vec<E::G1>,
    /// projection quotients of b_lagrange
    pub q_b: Vec<E::G1>,
    /// The `A` element in `G1`.
    pub a_g1: E::G1Affine,
    /// The `B` element in `G2`.
    pub b_g1: E::G1Affine,
    /// The `B` element in `G2`.
    pub b_g2: E::G2Affine,
    /// The `C` element in `G1`.
    pub c_g1: E::G1Affine,
}
