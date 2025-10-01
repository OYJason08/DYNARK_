#[cfg(feature = "parallel")]
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::{PrimeField, Zero};
use ark_relations::r1cs::{ConstraintMatrices, SynthesisError};
use ark_std::{cfg_iter, vec::Vec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

///faster getting lagrangian
pub fn improved_lagrange_sparse_get<F: PrimeField>(
    a_cols: &Vec<Vec<(usize, F)>>,
    b_cols: &Vec<Vec<(usize, F)>>,
    sparse_vec: &[(usize, F)],
) -> Result<(Vec<(usize, F)>, Vec<(usize, F)>), SynthesisError> {
    let mut a_pairs: Vec<(usize, F)>  = sparse_vec.par_iter().map(|(j,vl)|{
        a_cols[*j].par_iter().map(|(i, coef)| {(*i, *coef * *vl)}).collect::<Vec<(usize, F)>>()
    }).collect::<Vec<Vec<(usize, F)>>>().into_par_iter().flatten().collect::<Vec<(usize, F)>>();

    let mut b_pairs: Vec<(usize, F)>  = sparse_vec.par_iter().map(|(j,vl)|{
        b_cols[*j].par_iter().map(|(i, coef)| {(*i, *coef * *vl)}).collect::<Vec<(usize, F)>>()
    }).collect::<Vec<Vec<(usize, F)>>>().into_par_iter().flatten().collect::<Vec<(usize, F)>>();

    let mut a_lagrange_sparse: Vec<(usize, F)> = Vec::new();
    let mut b_lagrange_sparse: Vec<(usize, F)> = Vec::new();

    a_pairs.sort_by_key(|k| k.0);
    b_pairs.sort_by_key(|k| k.0);

    for (i, vl) in a_pairs {
        if let Some((last_i, sum)) = a_lagrange_sparse.last_mut() {
            if *last_i == i {
                *sum += vl; // accumulate
                continue;
            }
        }
        a_lagrange_sparse.push((i, vl)); // start new group
    }
    for (i, vl) in b_pairs {
        if let Some((last_i, sum)) = b_lagrange_sparse.last_mut() {
            if *last_i == i {
                *sum += vl; // accumulate
                continue;
            }
        }
        b_lagrange_sparse.push((i, vl)); // start new group
    }

    Ok((a_lagrange_sparse, b_lagrange_sparse))
}



///
#[inline]
pub fn lagrange_of_dense_vec<F: PrimeField>(
    matrices: &ConstraintMatrices<F>,
    dense_vec: &[F],
) -> Result<(Vec<F>, Vec<F>, Vec<F>), SynthesisError> {
    let num_constraints = matrices.num_constraints;
    let domain_size = num_constraints.next_power_of_two();

    // a matrix and z product
    let mut a_lagrg: Vec<F> = vec![F::zero(); domain_size];
    let mut b_lagrg = vec![F::zero(); domain_size];
    let mut c_lagrg = vec![F::zero(); domain_size];

    cfg_iter_mut!(a_lagrg[..num_constraints])
        .zip(cfg_iter!(&matrices.a))
        .for_each(|(a_lagrg_i, a_row)| {
            *a_lagrg_i = inner_product_sparse_f_dense_f(&a_row, dense_vec);
        });

    cfg_iter_mut!(b_lagrg[..num_constraints])
        .zip(cfg_iter!(&matrices.b))
        .for_each(|(b_lagrg_i, b_row)| {
            *b_lagrg_i = inner_product_sparse_f_dense_f(&b_row, dense_vec);
        });

    cfg_iter_mut!(c_lagrg[..num_constraints])
        .zip(cfg_iter!(&matrices.c))
        .for_each(|(c_lagrg_i, c_row)| {
            *c_lagrg_i = inner_product_sparse_f_dense_f(&c_row, dense_vec);
        });

    Ok((a_lagrg, b_lagrg, c_lagrg))
}

///
pub fn inner_product_sparse_f_dense_f<F: PrimeField>(terms: &[(F, usize)], assignment: &[F]) -> F {
    #[cfg(feature = "parallel")]
    if terms.len() < 5 {
        serial_inner_product_sparse_f_dense_f(terms, assignment)
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
    serial_inner_product_sparse_f_dense_f(terms, assignment)
}

fn serial_inner_product_sparse_f_dense_f<F: PrimeField>(
    terms: &[(F, usize)],
    assignment: &[F],
) -> F {
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

///
#[inline]
pub fn inner_product_sparse_f_dense_g<E: Pairing>(
    sparse_vec: &[(usize, E::ScalarField)],
    dense_vec: &[E::G1],
) -> E::G1 {
    #[cfg(feature = "parallel")]
    {

        if sparse_vec.len() < 5 {
            serial_inner_product_sparse_f_dense_g::<E>(sparse_vec, dense_vec)
        } else {
            sparse_vec
                .par_iter()
                .map(|(idx, coeff)| {
                    let val = dense_vec[*idx];
                    val * coeff
                })
                .reduce(|| E::G1::zero(), |a, b| a + b)
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        serial_inner_product_sparse_f_dense_g(sparse_vec, dense_vec)
    }
}

///
#[inline]
pub fn serial_inner_product_sparse_f_dense_g<E: Pairing>(
    sparse_vec: &[(usize, E::ScalarField)],
    dense_vec: &[E::G1],
) -> E::G1 {
    let mut sum = E::G1::zero();
    for (idx, coeff) in sparse_vec {
        let val = dense_vec[*idx];
        // sum += if coeff.is_one() { val } else {  val * coeff };
        sum += val * coeff;
    }
    sum
}