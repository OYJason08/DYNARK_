use ark_ec::pairing::Pairing;
use ark_ff::{Field, UniformRand};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::SynthesisError;
use ark_serialize::*;
use ark_std::{
    convert::TryInto,
    log2,
    rand::RngCore,
    vec::Vec,
};
use core::{mem::swap, time::Duration};
use rayon::prelude::*;
use std::{fmt::Debug, time::Instant};

// use ark_poly::EvaluationDomain;
// F: PrimeField, D: EvaluationDomain<PrimeField>

/// Number of Threads
pub const N_THREADS: usize = 1;
/// Size of Step of Sqrt FFT
pub const STEP_SQRT_FFT: usize = 1;
/// Size of Step of Transpose
pub const STEP_SQRT_TRANSPOSE: usize = 2;
/// Size of Step of Transpose
pub const STEP_SQRT_COPY: usize = 32;

#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize, Default)]
/// Stores info of FFT, including powers of generators and bit-reversion table
pub struct CustomParaFFT<E: Pairing> {
    /// size of a FFT array
    pub log_n: usize,
    /// size of a FFT array
    pub n: usize,
    /// power of roots of unity
    pub omegas: Vec<E::ScalarField>,
    /// power of inversion of roots of unity
    pub omega_invs: Vec<E::ScalarField>,
    /// list of bit-reversed match
    pub bit_rev_table: Vec<usize>,
    /// inverse of N
    pub bit_rev_task: Vec<usize>,
    /// Need Swap
    pub inv_n: E::ScalarField,
    /// covering_length per Thread
    pub len_thread: usize,
}

impl<E: Pairing> CustomParaFFT<E> {
    /// Create new FFT parameter header
    pub fn new(g_size: usize, g_gen: E::ScalarField) -> Self {
        let n = g_size;
        // let n = domain.size();
        if !n.is_power_of_two() {
            // panic!("Get N = {:?} not Power of Two", n);
        }
        if !N_THREADS.is_power_of_two() {
            panic!("Number of Threads = {:?} not Power of Two", N_THREADS);
        }
        if (N_THREADS << 1) > n {
            panic!(
                "Number of Threads = {:?}, its twice even more than N( {:?} )",
                N_THREADS, n
            );
        }

        let omega = g_gen;
        // let omega = domain.group_gen();
        // let omega = F::get_root_of_unity(n);

        let omega_inv = omega.inverse().unwrap();

        let log_n = log2(n) as usize;
        let f_one = <E::ScalarField as Field>::ONE;
        let f_two = f_one + f_one;

        let mut omegas = vec![f_one; n];
        let mut omega_invs = vec![f_one; n];

        for i in 1..n {
            omegas[i] = omegas[i - 1] * omega;
            omega_invs[i] = omega_invs[i - 1] * omega_inv;
        }

        let mut bit_rev_table = vec![0; n];
        let mut bit_rev_task: Vec<usize> = Vec::new();
        for i in 0..n {
            bit_rev_table[i] = (bit_rev_table[i >> 1] >> 1) | ((i & 1) << (log_n - 1));
            if i < bit_rev_table[i] {
                bit_rev_task.push(i);
                bit_rev_task.push(bit_rev_table[i]);
            }
        }

        if (bit_rev_task.len() & ((N_THREADS << 1) - 1)) > 0 {
            // panic!("Number of Threads = {:?}, Not Dividing N_bit_rev( {:?}
            // )", (N_THREADS<<1),bit_rev_task.len());
        }

        let mut inv_n = f_one;
        for _ in 0..log_n {
            inv_n = inv_n * f_two.inverse().unwrap();
        }
        let len_thread = n / N_THREADS;
        Self {
            log_n,
            n,
            omegas,
            omega_invs,
            bit_rev_table,
            bit_rev_task,
            inv_n,
            len_thread,
        }
    }

    ///  FFT on Field
    #[inline]
    pub fn f_butterfly(
        &self,
        a: &mut E::ScalarField,
        b: &mut E::ScalarField,
        i: usize,
        lev: usize,
        rev: bool,
    ) {
        let omega_pw = if !rev {
            self.omegas[i * ((self.n >> lev) >> 1)]
        } else {
            self.omega_invs[i * ((self.n >> lev) >> 1)]
        };
        let t = *b * omega_pw;
        *b = *a - t;
        *a += t;
    }

    ///  FFT on Group 1
    #[inline]
    pub fn g1_butterfly(&self, a: &mut E::G1, b: &mut E::G1, i: usize, lev: usize, rev: bool) {
        let omega_pw = if !rev {
            self.omegas[i * ((self.n >> lev) >> 1)]
        } else {
            self.omega_invs[i * ((self.n >> lev) >> 1)]
        };
        let t = *b * omega_pw;
        *b = *a - t;
        *a += t;
    }

    ///  FFT on Group 2
    #[inline]
    pub fn g2_butterfly(&self, a: &mut E::G2, b: &mut E::G2, i: usize, lev: usize, rev: bool) {
        let omega_pw = if !rev {
            self.omegas[i * ((self.n >> lev) >> 1)]
        } else {
            self.omega_invs[i * ((self.n >> lev) >> 1)]
        };
        let t = *b * omega_pw;
        *b = *a - t;
        *a += t;
    }
}

/// sqrt FFT
pub fn sqrt_fft<E: Pairing>(fft_info: &CustomParaFFT<E>, x: &mut FFTVectorTypes<E>, is_rev: bool) {
    // println!("It's really happening!");
    // if fft_info.log_n & 1 != 0 {
    //     panic!("FFT log_n Not Even");
    // }
    let rows_log = fft_info.log_n >> 1;
    let cols_log = fft_info.log_n - rows_log;
    let rows_cnt:usize = 1 << rows_log;
    let cols_cnt:usize = 1 << cols_log;
    type D<F> = GeneralEvaluationDomain<F>;
    let rows_domain: GeneralEvaluationDomain<<E as Pairing>::ScalarField> = D::new(rows_cnt)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();

    let cols_domain: GeneralEvaluationDomain<<E as Pairing>::ScalarField> = D::new(cols_cnt)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    match x {
        FFTVectorTypes::XF(ref mut xf) => {
            // First Transpose
            let copy = xf.clone();
            let cover_copy: Vec<E::ScalarField> = (0..fft_info.n)
                .into_par_iter()
                .map(|i: usize| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xf.copy_from_slice(&cover_copy);
            // Column FFT
            xf.par_chunks_mut(rows_cnt).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    rows_domain.ifft_in_place(&mut row_copy);
                } else {
                    rows_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Transpose with Variable Twiddle Factors
            let copy: Vec<E::ScalarField> = xf.clone();
            let cover_copy: Vec<E::ScalarField> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> cols_log;
                    let c = i & ((1 << cols_log) - 1);
                    copy[(c << rows_log) | r]
                        * if is_rev {
                            fft_info.omega_invs[r * c]
                        } else {
                            fft_info.omegas[r * c]
                        }
                })
                .collect();
            xf.copy_from_slice(&cover_copy);
            // Row FFT
            xf.par_chunks_mut(1 << cols_log).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    cols_domain.ifft_in_place(&mut row_copy);
                } else {
                    cols_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Final Transpose
            let copy: Vec<E::ScalarField> = xf.clone();
            let cover_copy: Vec<E::ScalarField> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xf.copy_from_slice(&cover_copy);
        },
        FFTVectorTypes::XG1(ref mut xg1) => {
            // First Transpose
            let copy = xg1.clone();
            let cover_copy: Vec<E::G1> = (0..fft_info.n)
                .into_par_iter()
                .map(|i: usize| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xg1.copy_from_slice(&cover_copy);
            // Column FFT
            xg1.par_chunks_mut(rows_cnt).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    rows_domain.ifft_in_place(&mut row_copy);
                } else {
                    rows_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Transpose with Variable Twiddle Factors
            let copy: Vec<E::G1> = xg1.clone();
            let cover_copy: Vec<E::G1> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> cols_log;
                    let c = i & ((1 << cols_log) - 1);
                    copy[(c << rows_log) | r]
                        * if is_rev {
                            fft_info.omega_invs[r * c]
                        } else {
                            fft_info.omegas[r * c]
                        }
                })
                .collect();
            xg1.copy_from_slice(&cover_copy);
            // Row FFT
            xg1.par_chunks_mut(1 << cols_log).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    cols_domain.ifft_in_place(&mut row_copy);
                } else {
                    cols_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Final Transpose
            let copy: Vec<E::G1> = xg1.clone();
            let cover_copy: Vec<E::G1> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xg1.copy_from_slice(&cover_copy);
        },
        FFTVectorTypes::XG2(ref mut xg2) => {
            // First Transpose
            let copy = xg2.clone();
            let cover_copy: Vec<E::G2> = (0..fft_info.n)
                .into_par_iter()
                .map(|i: usize| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xg2.copy_from_slice(&cover_copy);
            // Column FFT
            xg2.par_chunks_mut(rows_cnt).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    rows_domain.ifft_in_place(&mut row_copy);
                } else {
                    rows_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Transpose with Variable Twiddle Factors
            let copy: Vec<E::G2> = xg2.clone();
            let cover_copy: Vec<E::G2> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> cols_log;
                    let c = i & ((1 << cols_log) - 1);
                    copy[(c << rows_log) | r]
                        * if is_rev {
                            fft_info.omega_invs[r * c]
                        } else {
                            fft_info.omegas[r * c]
                        }
                })
                .collect();
            xg2.copy_from_slice(&cover_copy);
            // Row FFT
            xg2.par_chunks_mut(1 << cols_log).for_each(|row| {
                let mut row_copy = row.to_vec();
                if is_rev {
                    cols_domain.ifft_in_place(&mut row_copy);
                } else {
                    cols_domain.fft_in_place(&mut row_copy);
                }
                row.copy_from_slice(&row_copy);
            });
            // Final Transpose
            let copy: Vec<E::G2> = xg2.clone();
            let cover_copy: Vec<E::G2> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> rows_log;
                    let c = i & ((1 << rows_log) - 1);
                    copy[(c << cols_log) | r]
                })
                .collect();
            xg2.copy_from_slice(&cover_copy);
        },
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
/// Describe the phase of sqrt FFT process
enum FftPhase {
    /// first transpose
    PreTranspose,
    /// phase1
    ColFFT,
    /// phase2
    TRANSPOSE,
    /// phase3
    RowFFT,
    /// final transpose
    SufTranspose
}

#[derive(Clone, Debug, PartialEq)]
/// Describe the status of sqrt FFT process
pub struct SqrtFftStatus {
    /// Whether is reverse of fft
    is_rev: bool,
    /// Current Phase
    phase: FftPhase,
    /// Current Level of phase
    lev: usize,
}
impl SqrtFftStatus{
    pub fn new(is_rev: bool)->Self{
        Self{
            is_rev,
            phase:FftPhase::PreTranspose,
            lev:0
        }
    }
}



/// step forward sqrt FFT
pub fn sqrt_fft_fw<E: Pairing>(
    // sub_domain: &GeneralEvaluationDomain<<E as Pairing>::ScalarField>,
    haf_info: &CustomParaFFT<E>,
    sqrt_fft_info: &CustomParaFFT<E>,
    x: &mut FFTVectorTypes<E>,
    status: &mut SqrtFftStatus,
    mem_for_transpose: &mut FFTVectorTypes<E>,
) -> bool {
    let is_rev = status.is_rev;
    let haf_log = sqrt_fft_info.log_n >> 1;
    let mx_step: usize = 1<<haf_log;
    // println!("X {:?}, ", x);
    // println!("Status {:?}    ", status);
    match status.phase {
        FftPhase::PreTranspose | FftPhase::SufTranspose =>{
            let stp = std::cmp::min(mx_step, STEP_SQRT_COPY);
            if status.lev < (1<< haf_log){
                let l = status.lev  << haf_log;
                let r = (status.lev+ stp) << haf_log;
                match x {
                    FFTVectorTypes::XF(ref mut xf) => {
                        if let FFTVectorTypes::XF(ref mut copy) = mem_for_transpose{
                            (l..r).into_par_iter().zip(copy[l..r].par_iter_mut()).for_each(|(i,v)|{
                                let r = i >> haf_log;
                                let c = i&((1<<haf_log)-1);
                                *v = xf[(c << haf_log) | r];
                            })
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                    FFTVectorTypes::XG1(ref mut xg1) => {
                        if let FFTVectorTypes::XG1(ref mut copy) = mem_for_transpose{
                            (l..r).into_par_iter().zip(copy[l..r].par_iter_mut()).for_each(|(i,v)|{
                                let r = i >> haf_log;
                                let c = i&((1<<haf_log)-1);
                                *v = xg1[(c << haf_log) | r];
                            })
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                    FFTVectorTypes::XG2(ref mut xg2) => {
                        if let FFTVectorTypes::XG2(ref mut copy) = mem_for_transpose{
                            (l..r).into_par_iter().zip(copy[l..r].par_iter_mut()).for_each(|(i,v)|{
                                let r = i >> haf_log;
                                let c = i&((1<<haf_log)-1);
                                *v = xg2[(c << haf_log) | r];
                            })
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    }
                }
            }
            else{
                let l = (status.lev-(1<<haf_log))  << haf_log;
                let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                match x {
                    FFTVectorTypes::XF(ref mut xf) => {
                        if let FFTVectorTypes::XF(ref mut copy) = mem_for_transpose{
                            xf[l..r].copy_from_slice(&copy[l..r]);
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                    FFTVectorTypes::XG1(ref mut xg1) => {
                        if let FFTVectorTypes::XG1(ref mut copy) = mem_for_transpose{
                            xg1[l..r].copy_from_slice(&copy[l..r]);
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                    FFTVectorTypes::XG2(ref mut xg2) => {
                        if let FFTVectorTypes::XG2(ref mut copy) = mem_for_transpose{
                            xg2[l..r].copy_from_slice(&copy[l..r]);
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    }
                }
            }
            status.lev += stp;
            if status.lev >= 2<<haf_log{
                if status.phase == FftPhase::PreTranspose{
                    status.lev = 0;
                    status.phase = FftPhase::ColFFT;
                }
                else{
                    status.lev = 0;
                    status.phase = FftPhase::PreTranspose;
                    return true;
                }
            }
        }
        FftPhase::ColFFT => {
            match x {
                FFTVectorTypes::XF(ref mut xf) => {
                    xf[(status.lev << haf_log)..((status.lev + STEP_SQRT_FFT) << haf_log)]
                        .par_chunks_mut(1 << haf_log)
                        .for_each(|row| {
                            let mut xf = FFTVectorTypes::XF(row.to_vec());
                            sqrt_fft(&haf_info, &mut xf, is_rev);
                            if let FFTVectorTypes::XF(v) = xf {
                                row.copy_from_slice(&v);
                            }
                        });
                },
                FFTVectorTypes::XG1(ref mut xg1) => {
                    xg1[(status.lev << haf_log)..((status.lev + STEP_SQRT_FFT) << haf_log)]
                        .par_chunks_mut(1 << haf_log)
                        .for_each(|row| {
                            let mut xg1 = FFTVectorTypes::XG1(row.to_vec());
                            sqrt_fft(&haf_info, &mut xg1, is_rev);
                            if let FFTVectorTypes::XG1(v) = xg1 {
                                row.copy_from_slice(&v);
                            }
                        });
                },
                FFTVectorTypes::XG2(ref mut xg2) => {
                    xg2[(status.lev << haf_log)..((status.lev + STEP_SQRT_FFT) << haf_log)]
                        .par_chunks_mut(1 << haf_log)
                        .for_each(|row| {
                            let mut xg2 = FFTVectorTypes::XG2(row.to_vec());
                            sqrt_fft(&haf_info, &mut xg2, is_rev);
                            if let FFTVectorTypes::XG2(v) = xg2 {
                                row.copy_from_slice(&v);
                            }
                        });
                },
            }
            status.lev += STEP_SQRT_FFT;
            if status.lev >= 1 << haf_log {
                status.phase = FftPhase::TRANSPOSE;
                status.lev = 0;
            }
        },
        FftPhase::TRANSPOSE => {
            if status.lev < (1<< haf_log){
                let stp = std::cmp::min(mx_step, STEP_SQRT_TRANSPOSE);
                let l = status.lev  << haf_log;
                let r = (status.lev+ stp) << haf_log;
                match x {
                    FFTVectorTypes::XF(ref mut xf) => {
                        if let FFTVectorTypes::XF(ref mut copy) = mem_for_transpose{
                            (l..r).into_par_iter().zip(copy[l..r].par_iter_mut()).for_each(|(i,v)|{
                                let r = i >> haf_log;
                                let c = i&((1<<haf_log)-1);
                                *v = xf[(c << haf_log) | r] * 
                                    if status.is_rev {
                                        sqrt_fft_info.omega_invs[r * c]
                                    } else {
                                        sqrt_fft_info.omegas[r * c]
                                    }
                            })
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                    FFTVectorTypes::XG1(ref mut xg1) => {
                        if let FFTVectorTypes::XG1(ref mut copy) = mem_for_transpose{
                            (l..r).into_par_iter().zip(copy[l..r].par_iter_mut()).for_each(|(i,v)|{
                                let r = i >> haf_log;
                                let c = i&((1<<haf_log)-1);
                                if r * c > (1<<(haf_log<<1)){

                                    println!("i = {:?}, r = {:?}  c = {:?}",i,r,c);
                                }
                                *v = xg1[(c << haf_log) | r] * 
                                    if status.is_rev {
                                        sqrt_fft_info.omega_invs[r * c]
                                    } else {
                                        sqrt_fft_info.omegas[r * c]
                                    }
                            })
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                    FFTVectorTypes::XG2(ref mut xg2) => {
                        if let FFTVectorTypes::XG2(ref mut copy) = mem_for_transpose{
                            (l..r).into_par_iter().zip(copy[l..r].par_iter_mut()).for_each(|(i,v)|{
                                let r = i >> haf_log;
                                let c = i&((1<<haf_log)-1);
                                *v = xg2[(c << haf_log) | r] * 
                                    if status.is_rev {
                                        sqrt_fft_info.omega_invs[r * c]
                                    } else {
                                        sqrt_fft_info.omegas[r * c]
                                    }
                            })
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                }
                status.lev += stp;
            }
            else{
                let stp = std::cmp::min(mx_step, STEP_SQRT_COPY);
                let l = (status.lev-(1<<haf_log))  << haf_log;
                let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                match x {
                    FFTVectorTypes::XF(ref mut xf) => {
                        if let FFTVectorTypes::XF(ref mut copy) = mem_for_transpose{
                            xf[l..r].copy_from_slice(&copy[l..r]);
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                    FFTVectorTypes::XG1(ref mut xg1) => {
                        if let FFTVectorTypes::XG1(ref mut copy) = mem_for_transpose{
                            xg1[l..r].copy_from_slice(&copy[l..r]);
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                    FFTVectorTypes::XG2(ref mut xg2) => {
                        if let FFTVectorTypes::XG2(ref mut copy) = mem_for_transpose{
                            xg2[l..r].copy_from_slice(&copy[l..r]);
                        }
                        else{
                            panic!("Feeding Wrong memory");
                        }
                    },
                }
                status.lev += stp;
                if status.lev >= 2 << haf_log {
                    status.phase = FftPhase::RowFFT;
                    status.lev = 0;
                }
            }
            // status.phase = FftPhase::RowFFT;
            // status.lev = 0;
            // transpose_mult(&sqrt_fft_info, x, status.is_rev);
        },
        FftPhase::RowFFT => {
            match x {
                FFTVectorTypes::XF(ref mut xf) => {
                    xf[(status.lev << haf_log)..((status.lev + STEP_SQRT_FFT) << haf_log)]
                        .par_chunks_mut(1 << haf_log)
                        .for_each(|row| {
                            let mut xf = FFTVectorTypes::XF(row.to_vec());
                            sqrt_fft(&haf_info, &mut xf, is_rev);
                            if let FFTVectorTypes::XF(v) = xf {
                                row.copy_from_slice(&v);
                            }
                        });
                },
                FFTVectorTypes::XG1(ref mut xg1) => {
                    xg1[(status.lev << haf_log)..((status.lev + STEP_SQRT_FFT) << haf_log)]
                        .par_chunks_mut(1 << haf_log)
                        .for_each(|row| {
                            let mut xg1 = FFTVectorTypes::XG1(row.to_vec());
                            sqrt_fft(&haf_info, &mut xg1, is_rev);
                            if let FFTVectorTypes::XG1(v) = xg1 {
                                row.copy_from_slice(&v);
                            }
                        });
                },
                FFTVectorTypes::XG2(ref mut xg2) => {
                    xg2[(status.lev << haf_log)..((status.lev + STEP_SQRT_FFT) << haf_log)]
                        .par_chunks_mut(1 << haf_log)
                        .for_each(|row| {
                            let mut xg2 = FFTVectorTypes::XG2(row.to_vec());
                            sqrt_fft(&haf_info, &mut xg2, is_rev);
                            if let FFTVectorTypes::XG2(v) = xg2 {
                                row.copy_from_slice(&v);
                            }
                        });
                },
            }
            status.lev += STEP_SQRT_FFT;
            if status.lev >= 1 << haf_log {
                status.phase = FftPhase::SufTranspose;
                status.lev = 0;
            }
        },
    }
    return false;
}

/// Step by Step Sqrt FFT
pub fn sbs_sqrt_fft<E: Pairing>(
    fft_info: &CustomParaFFT<E>,
    x: &mut FFTVectorTypes<E>,
    is_rev: bool,
) {
    if fft_info.log_n & 1 != 0 {
        panic!("FFT log_n Not Even");
    }
    let haf_log = fft_info.log_n >> 1;
    let sub_domain_size: usize = (1usize) << haf_log;
    type D<F> = GeneralEvaluationDomain<F>;
    let sub_domain: GeneralEvaluationDomain<<E as Pairing>::ScalarField> = D::new(sub_domain_size)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    let sub_info = CustomParaFFT::<E>::new(sub_domain_size, sub_domain.group_gen());
    let mut status = SqrtFftStatus {
        is_rev,
        phase: FftPhase::PreTranspose,
        lev: 0,
    };
    let mut mx_dur_transpose = Duration::ZERO;
    let mut mx_dur_sqrt_fft = Duration::ZERO;
    let mut mem_for_traspose = x.clone();
    loop {
        let start1 = Instant::now();
        let step_type = status.phase;
        let finish = sqrt_fft_fw::<E>(
            // &sub_domain, 
            &sub_info,
            &fft_info, 
            x, 
            &mut status,
            &mut mem_for_traspose
        );
        let duration1 = start1.elapsed();
        if matches!(step_type, FftPhase::TRANSPOSE) || matches!(step_type, FftPhase::PreTranspose) || matches!(step_type, FftPhase::SufTranspose){
            if duration1 > mx_dur_transpose {
                mx_dur_transpose = duration1;
            }
        } else {
            if duration1 > mx_dur_sqrt_fft {
                mx_dur_sqrt_fft = duration1;
            }
        }
        if finish {
            break;
        }
    }
    println!("Max_SqrtFFT time :  {:?}", mx_dur_sqrt_fft);
    println!("Max_Transpose time :  {:?}", mx_dur_transpose);
}

/// Different types of Element doing fft
#[derive(Clone, Debug, PartialEq)]
pub enum FFTVectorTypes<E: Pairing> {
    XF(Vec<E::ScalarField>),
    XG1(Vec<E::G1>),
    XG2(Vec<E::G2>),
}

/// parse vector into array
pub fn vec_to_arr(v: Vec<usize>) -> [usize; N_THREADS << 1] {
    v.try_into().unwrap_or_else(|v: Vec<usize>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            N_THREADS << 1,
            v.len()
        )
    })
}

// Transpose a matrix stored in a vector
pub fn transpose<E: Pairing>(fft_info: &CustomParaFFT<E>, x: &mut FFTVectorTypes<E>) {
    let haf_log = fft_info.log_n >> 1;
    if fft_info.log_n & 1 == 1 {
        panic!("Transpose not support odd log_n");
    }
    match x {
        FFTVectorTypes::XF(ref mut xf) => {
            // First Transpose
            let copy = xf.clone();
            let cover_copy: Vec<E::ScalarField> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> haf_log;
                    let c = i & ((1 << haf_log) - 1);
                    copy[(c << haf_log) | r]
                })
                .collect();
            xf.copy_from_slice(&cover_copy);
        },
        FFTVectorTypes::XG1(ref mut xg1) => {
            // First Transpose
            let copy = xg1.clone();
            let cover_copy: Vec<E::G1> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> haf_log;
                    let c = i & ((1 << haf_log) - 1);
                    copy[(c << haf_log) | r]
                })
                .collect();
            xg1.copy_from_slice(&cover_copy);
        },
        FFTVectorTypes::XG2(ref mut xg2) => {
            // First Transpose
            let copy = xg2.clone();
            let cover_copy: Vec<E::G2> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> haf_log;
                    let c = i & ((1 << haf_log) - 1);
                    copy[(c << haf_log) | r]
                })
                .collect();
            xg2.copy_from_slice(&cover_copy);
        },
    }
}

/// Transpose Mult
pub fn transpose_mult<E: Pairing>(fft_info: &CustomParaFFT<E>, x: &mut FFTVectorTypes<E>, is_rev: bool) {
    let haf_log = fft_info.log_n >> 1;
    if fft_info.log_n & 1 == 1 {
        panic!("Transpose not support odd log_n");
    }
    match x {
        FFTVectorTypes::XF(ref mut xf) => {
            // First Transpose
            let copy = xf.clone();
            let cover_copy: Vec<E::ScalarField> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> haf_log;
                    let c = i & ((1 << haf_log) - 1);
                    copy[(c << haf_log) | r] * 
                        if is_rev {
                            fft_info.omega_invs[r*c]
                        } else {
                            fft_info.omegas[r*c]
                        }
                })
                .collect();
            xf.copy_from_slice(&cover_copy);
        },
        FFTVectorTypes::XG1(ref mut xg1) => {
            // First Transpose
            let copy = xg1.clone();
            let cover_copy: Vec<E::G1> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> haf_log;
                    let c = i & ((1 << haf_log) - 1);
                    copy[(c << haf_log) | r] * 
                        if is_rev {
                            fft_info.omega_invs[r*c]
                        } else {
                            fft_info.omegas[r*c]
                        }
                })
                .collect();
            xg1.copy_from_slice(&cover_copy);
        },
        FFTVectorTypes::XG2(ref mut xg2) => {
            // First Transpose
            let copy = xg2.clone();
            let cover_copy: Vec<E::G2> = (0..fft_info.n)
                .into_par_iter()
                .map(|i| {
                    let r = i >> haf_log;
                    let c = i & ((1 << haf_log) - 1);
                    copy[(c << haf_log) | r] * 
                        if is_rev {
                            fft_info.omega_invs[r*c]
                        } else {
                            fft_info.omegas[r*c]
                        }
                })
                .collect();
            xg2.copy_from_slice(&cover_copy);
        },
    }
}

/// Record a status of a FFT thread
#[derive(Clone, Debug, PartialEq)]
pub struct FftStatu {
    // /// Based on a given CustomParaFFT
    // pub fft_info: &CustomParaFFT<E: Pairing>,
    /// Record whether is the reversion
    pub is_rev: bool,
    /// Record current level
    pub lev: i32,
    /// Record current position
    pub id: usize,
    /// Record id of all Threads
    pub ids: Vec<usize>,
}

impl FftStatu {
    /// Create a mew FFT Statu
    pub fn new(is_rev: bool) -> Self {
        let lev = -1;
        let id = 0;
        let ids = vec![0; N_THREADS];
        Self {
            is_rev,
            lev,
            id,
            ids,
        }
    }
}

/// Do a step of fft
fn fft_foward<E: Pairing>(
    fft_info: &CustomParaFFT<E>,
    fft_stat: &mut FftStatu,
    x: &mut FFTVectorTypes<E>,
) -> bool {
    let id = fft_stat.id;
    let cur_lev = fft_stat.lev;
    let mut go_nxt_lev = false;
    let mut nxt_lev = fft_stat.lev;
    let mut nxt_id = fft_stat.id + N_THREADS;
    // println!("Lev = {},    id = {},    N_THREADS = {}", cur_lev, id, N_THREADS);
    if cur_lev < 0 {
        let tasks: Vec<usize> = (id..(id + (N_THREADS << 1)))
            .map(|j| fft_info.bit_rev_task[j])
            .collect::<Vec<usize>>();

        match x {
            FFTVectorTypes::XF(ref mut xf) => {
                let mut tsk = xf.get_disjoint_mut(vec_to_arr(tasks)).unwrap();
                tsk.par_chunks_mut(2).for_each(|chunk| {
                    let [xi, xj] = chunk.get_disjoint_mut([0, 1]).unwrap();
                    swap(*xi, *xj);
                });
            },
            FFTVectorTypes::XG1(ref mut xg1) => {
                let mut tsk = xg1.get_disjoint_mut(vec_to_arr(tasks)).unwrap();
                tsk.par_chunks_mut(2).for_each(|chunk| {
                    let [xi, xj] = chunk.get_disjoint_mut([0, 1]).unwrap();
                    swap(*xi, *xj);
                });
            },
            FFTVectorTypes::XG2(ref mut xg2) => {
                let mut tsk = xg2.get_disjoint_mut(vec_to_arr(tasks)).unwrap();
                tsk.par_chunks_mut(2).for_each(|chunk| {
                    let [xi, xj] = chunk.get_disjoint_mut([0, 1]).unwrap();
                    swap(*xi, *xj);
                });
            },
        }
        nxt_id = fft_stat.id + (N_THREADS << 1);
        go_nxt_lev = nxt_id >= fft_info.bit_rev_task.len();
    } else if cur_lev < (fft_info.log_n as i32) {
        let tasks: Vec<usize> = (0..(N_THREADS << 1))
            .map(|j| fft_stat.ids[j >> 1] + (j & 1) * (1 << cur_lev))
            .collect::<Vec<usize>>();

        // println!("Lev = {}:  ,    Op Pos: {:?}     Task:  {:?}", cur_lev,
        // fft_stat.ids, tasks);
        match x {
            FFTVectorTypes::XF(ref mut xf) => {
                let mut tsk = xf.get_disjoint_mut(vec_to_arr(tasks)).unwrap();
                tsk.par_chunks_mut(2).enumerate().for_each(|(i, chunk)| {
                    let [xi, xj] = chunk.get_disjoint_mut([0, 1]).unwrap();
                    fft_info.f_butterfly(
                        xi,
                        xj,
                        fft_stat.ids[i] & ((1 << cur_lev) - 1),
                        cur_lev as usize,
                        fft_stat.is_rev,
                    );
                });
            },
            FFTVectorTypes::XG1(ref mut xg1) => {
                let mut tsk = xg1.get_disjoint_mut(vec_to_arr(tasks)).unwrap();
                tsk.par_chunks_mut(2).enumerate().for_each(|(i, chunk)| {
                    let [xi, xj] = chunk.get_disjoint_mut([0, 1]).unwrap();
                    fft_info.g1_butterfly(
                        xi,
                        xj,
                        fft_stat.ids[i] & ((1 << cur_lev) - 1),
                        cur_lev as usize,
                        fft_stat.is_rev,
                    );
                });
            },
            FFTVectorTypes::XG2(ref mut xg2) => {
                let mut tsk = xg2.get_disjoint_mut(vec_to_arr(tasks)).unwrap();
                tsk.par_chunks_mut(2).enumerate().for_each(|(i, chunk)| {
                    let [xi, xj] = chunk.get_disjoint_mut([0, 1]).unwrap();
                    fft_info.g2_butterfly(
                        xi,
                        xj,
                        fft_stat.ids[i] & ((1 << cur_lev) - 1),
                        cur_lev as usize,
                        fft_stat.is_rev,
                    );
                });
            },
        }
        fft_stat.ids.par_iter_mut().for_each(|i| {
            *i = *i + 1;
            if ((*i >> cur_lev) & 1) > 0 {
                *i += 1 << cur_lev;
            }
        });
        if let Some(last_id) = fft_stat.ids.last() {
            go_nxt_lev = last_id >= &fft_info.n;
        }
    } else {
        // println!("Mul Rev  {:?}   while id = {:?}\n",i, id);
        match x {
            FFTVectorTypes::XF(ref mut xf) => {
                let seg = &mut xf[id..id + N_THREADS];
                seg.par_iter_mut().for_each(|x| *x *= fft_info.inv_n);
            },
            FFTVectorTypes::XG1(ref mut xg1) => {
                let seg = &mut xg1[id..id + N_THREADS];
                seg.par_iter_mut().for_each(|x| *x *= fft_info.inv_n);
            },
            FFTVectorTypes::XG2(ref mut xg2) => {
                let seg = &mut xg2[id..id + N_THREADS];
                seg.par_iter_mut().for_each(|x| *x *= fft_info.inv_n);
            },
        }
        go_nxt_lev = nxt_id >= fft_info.n;
    }
    // finished processing this step, now prepare for the next step
    if go_nxt_lev {
        nxt_id = 0;
        nxt_lev = nxt_lev + 1;
        // println!("Go Next:  nxt_lev = {},  nxt_id = {}", nxt_lev, nxt_id);
        let log_n = fft_info.log_n;
        if nxt_lev > (log_n as i32) {
            fft_stat.lev = -1;
            fft_stat.id = 0;
            return true;
        }
        if nxt_lev >= (log_n as i32) && !fft_stat.is_rev {
            fft_stat.lev = -1;
            fft_stat.id = 0;
            return true;
        }
        if 0 <= nxt_lev && nxt_lev < (log_n as i32) {
            let len_thread = fft_info.len_thread;
            if (len_thread >> (nxt_lev + 1)) > 0 {
                fft_stat.ids = (0..N_THREADS)
                    .map(|j: usize| j * len_thread)
                    .collect::<Vec<usize>>();
                // println!("prepare:  Lev = {:?},  CASE 1,  ids = {:?} \n",
                // nxt_lev, fft_stat.ids);
            } else {
                let per_seg: usize = N_THREADS / (1 << ((log_n as i32) - (nxt_lev + 1)));
                fft_stat.ids = (0..N_THREADS)
                    .map(|j| {
                        ((j / per_seg) << (nxt_lev + 1)) + ((j % per_seg) << nxt_lev) / per_seg
                    })
                    .collect::<Vec<usize>>();
                // println!("prepare:  Lev = {:?},  CASE 2,  ids = {:?} \n",
                // nxt_lev, fft_stat.ids);
            }
        }
        fft_stat.lev = nxt_lev;
    }
    fft_stat.id = nxt_id;
    return false;
}

/// Once-All-Alone FFT
pub fn par_fft<E: Pairing>(fft_info: &CustomParaFFT<E>, x: &mut FFTVectorTypes<E>, is_rev: bool) {
    let mut fft_stat: FftStatu = FftStatu::new(is_rev);

    let start1 = Instant::now();
    let start2 = Instant::now();
    let mut lev1 = false;
    let mut lev2 = false;

    loop {
        let finished = fft_foward(&fft_info, &mut fft_stat, x);
        // println!("{:?}",fft_stat);
        if !lev1 && fft_stat.lev >= 0 {
            let duration1 = start1.elapsed();
            lev1 = true;
            println!("Level -1 done by {:?}", duration1);
        }
        if !lev2 && lev1 && fft_stat.lev >= (fft_info.log_n as i32) {
            let duration2: core::time::Duration = start2.elapsed();
            lev2 = true;
            println!("Level logn done by {:?}", duration2);
        }
        if finished {
            break;
        }
    }
}

/// Test of FFT
pub fn fft_bench_test<E: Pairing, R: RngCore>(domain_size: usize, rng: &mut R) {
    type D<F> = GeneralEvaluationDomain<F>;
    let domain = D::new(domain_size)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    let fft_info: CustomParaFFT<E> = CustomParaFFT::<E>::new(domain_size, domain.group_gen());
    let n = fft_info.n;
    let x_f_raw: Vec<E::G1> = (0..n)
        .map(
            |i| E::G1::rand(rng), // if i==1 {E::ScalarField::ONE} else {E::ScalarField::zero()}
        )
        .collect();
    let mut x_f_copy = x_f_raw.clone();
    let start1 = Instant::now();
    domain.fft_in_place(&mut x_f_copy);
    let duration1 = start1.elapsed();
    println!("Base FFT:{:?}", duration1);
    // println!("Ans:\n{:?}\n", x_f_copy);
    // return;

    let mut x_f = FFTVectorTypes::<E>::XG1(x_f_raw);
    // transpose(&fft_info, &mut x_f);
    let start2 = Instant::now();
    sbs_sqrt_fft::<E>(&fft_info, &mut x_f, false);
    // sqrt_fft::<E>(&fft_info,&mut x_f, false);
    // Self::stepar_fft(&fft_info, &mut x_f, false);
    let duration2 = start2.elapsed();
    // transpose(&fft_info, &mut x_f);

    println!("Cstm FFT:{:?}", duration2);

    if let FFTVectorTypes::XG1(x_f_res) = x_f {
        if x_f_res == x_f_copy {
            println!("FFT Correct!");
        } else {
            println!("FFT Wrong!!!!!!!!!!!!!!");
        }
    } else {
        panic!("WTF");
    }
    let mut x_f_2 = FFTVectorTypes::<E>::XG1(x_f_copy.clone());
    domain.ifft_in_place(&mut x_f_copy);

    // transpose(&fft_info, &mut x_f_2);
    sbs_sqrt_fft::<E>(&fft_info, &mut x_f_2, true);
    // transpose(&fft_info, &mut x_f_2);

    if let FFTVectorTypes::XG1(x_f_res_2) = x_f_2 {
        if x_f_res_2 == x_f_copy {
            println!("iFFT Correct!");
        } else {
            println!("iFFT Wrong!!!!!!!!!!!!!!");
        }
    } else {
        panic!("WTF");
    }
}

///
pub fn fft_g<E: Pairing>(
    domain: &GeneralEvaluationDomain<<E as Pairing>::ScalarField>,
    x: &mut [E::G1Affine],
) -> Vec<E::G1Affine> {
    let mut res_g1: Vec<E::G1> = x
        .par_iter()
        .map(|p| (*p).into()) // or p.into() if you consume
        .collect();
    domain.fft_in_place(&mut res_g1);
    let res: Vec<E::G1Affine> = res_g1
        .par_iter()
        .map(|p| (*p).into()) // or p.into() if you consume
        .collect();
    return res;
}

///
#[inline]
pub fn convolution_f_g<E: Pairing>(a: &[E::ScalarField], b: &[E::G1]) -> Vec<<E as Pairing>::G1> {
    let a_len = a.len();
    let b_len = b.len();
    debug_assert_eq!(a_len, b_len);
    let domain = GeneralEvaluationDomain::<E::ScalarField>::new(a_len)
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    let a_fft = domain.fft(a);
    let mut b_fft: Vec<E::G1> = domain.fft(b);
    b_fft
        .par_iter_mut()
        .zip(a_fft.par_iter())
        .for_each(|(x_b, x_a)| {
            *x_b *= *x_a;
        });
    domain.ifft_in_place(&mut b_fft);
    b_fft
}

///
pub fn convolution_f_f<E: Pairing>(
    a: &[E::ScalarField],
    b: &[E::ScalarField],
) -> Vec<<E as Pairing>::ScalarField> {
    let a_len = a.len();
    let b_len = b.len();
    debug_assert_eq!(a_len, b_len);
    let domain = GeneralEvaluationDomain::<E::ScalarField>::new(a.len())
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)
        .unwrap();
    let mut a_fft = domain.fft(a);
    let b_fft = domain.fft(b);
    a_fft
        .par_iter_mut()
        .zip(b_fft.par_iter())
        .for_each(|(x_a, x_b)| {
            *x_a *= *x_b;
        });
    domain.ifft_in_place(&mut a_fft);
    a_fft
}

///
#[inline]
pub fn elementwise_product_f_f<E: Pairing>(
    a: &[E::ScalarField],
    b: &[E::ScalarField],
) -> Vec<<E as Pairing>::ScalarField> {
    a.iter().zip(b.iter()).map(|(a, b)| *b * a).collect()
}

///
#[inline]
pub fn elementwise_product_f_g<E: Pairing>(
    a: &[E::ScalarField],
    b: &[E::G1],
) -> Vec<<E as Pairing>::G1> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| *b * a)
        .collect()
}

///
#[inline]
pub fn elementwise_add_g_g<E: Pairing>(a: &[E::G1], b: &[E::G1]) -> Vec<<E as Pairing>::G1> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| *a + b)
        .collect()
}





/// a step of sbs FFT for field
pub fn sqrt_fft_fw_field<E: Pairing>(
    sub_domain: &GeneralEvaluationDomain<<E as Pairing>::ScalarField>,
    sqrt_fft_info: &CustomParaFFT<E>,
    xf: &mut Vec<E::ScalarField>,
    status: &mut SqrtFftStatus,
    mem_for_transpose: &mut Vec<E::ScalarField>,
    steps: usize,
) -> (bool, usize) {
    let is_rev = status.is_rev;
    let haf_log = sqrt_fft_info.log_n >> 1;
    let mx_step: usize = 1 << haf_log;
    let mut steps_rem = steps;
    while steps_rem >0{
        match status.phase {
            FftPhase::PreTranspose | FftPhase::SufTranspose =>{
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_COPY);
                if status.lev < (1<< haf_log){
                    let cnt_stp = std::cmp::min(steps_rem, (mx_step -status.lev) /unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = status.lev  << haf_log;
                    let r = (status.lev+ stp) << haf_log;
                    (l..r).into_par_iter().zip(mem_for_transpose[l..r].par_iter_mut()).for_each(|(i,v)|{
                        let r = i >> haf_log;
                        let c = i&((1<<haf_log)-1);
                        *v = xf[(c << haf_log) | r];
                    });
                    status.lev += stp;
                }
                else{
                    let cnt_stp = std::cmp::min(steps_rem, ((mx_step<<1) -status.lev) /unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = (status.lev-(1<<haf_log))  << haf_log;
                    let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                    xf[l..r].copy_from_slice(&mem_for_transpose[l..r]);
                    status.lev += stp;
                }
                if status.lev >= 2<<haf_log{
                    if status.phase == FftPhase::PreTranspose{
                        status.lev = 0;
                        status.phase = FftPhase::ColFFT;
                    }
                    else{
                        status.lev = 0;
                        status.phase = FftPhase::PreTranspose;
                        return (true, steps_rem);
                    }
                }
            }
            FftPhase::ColFFT => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_FFT);
                let cnt_stp = std::cmp::min(steps_rem, (mx_step-status.lev)/unit_stp);
                let stp = cnt_stp * unit_stp;
                steps_rem -= cnt_stp;
                let haf_info = CustomParaFFT::<E>::new(1 << haf_log, sub_domain.group_gen());
                xf[(status.lev << haf_log)..((status.lev + stp) << haf_log)]
                    .par_chunks_mut(1 << haf_log)
                    .for_each(|row| {
                        let mut xf = FFTVectorTypes::XF(row.to_vec());
                        sqrt_fft(&haf_info, &mut xf, is_rev);
                        if let FFTVectorTypes::XF(v) = xf {
                            row.copy_from_slice(&v);
                        }
                    });
                status.lev += stp;
                if status.lev >= 1 << haf_log {
                    status.phase = FftPhase::TRANSPOSE;
                    status.lev = 0;
                }
            },
            FftPhase::TRANSPOSE => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_TRANSPOSE);
                if status.lev < (1<< haf_log){
                    let cnt_stp = std::cmp::min(steps_rem, (mx_step-status.lev)/unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = status.lev  << haf_log;
                    let r = (status.lev+ stp) << haf_log;
                    (l..r).into_par_iter().zip(mem_for_transpose[l..r].par_iter_mut()).for_each(|(i,v)|{
                        let r = i >> haf_log;
                        let c = i&((1<<haf_log)-1);
                        *v = xf[(c << haf_log) | r] * 
                            if status.is_rev {
                                sqrt_fft_info.omega_invs[r * c]
                            } else {
                                sqrt_fft_info.omegas[r * c]
                            }
                    });
                    status.lev += stp;
                }
                else{
                    let cnt_stp = std::cmp::min(steps_rem, ((mx_step<<1)-status.lev)/unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = (status.lev-(1<<haf_log))  << haf_log;
                    let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                    xf[l..r].copy_from_slice(&mem_for_transpose[l..r]);
                    status.lev += stp;
                    if status.lev >= 2 << haf_log {
                        status.phase = FftPhase::RowFFT;
                        status.lev = 0;
                    }
                }
            },
            FftPhase::RowFFT => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_FFT);
                let cnt_stp = std::cmp::min(steps_rem, (mx_step - status.lev)/unit_stp);
                let stp = cnt_stp * unit_stp;
                steps_rem -= cnt_stp;
                let haf_info = CustomParaFFT::<E>::new(1 << haf_log, sub_domain.group_gen());
                xf[(status.lev << haf_log)..((status.lev + stp) << haf_log)]
                    .par_chunks_mut(1 << haf_log)
                    .for_each(|row| {
                        let mut xf = FFTVectorTypes::XF(row.to_vec());
                        sqrt_fft(&haf_info, &mut xf, is_rev);
                        if let FFTVectorTypes::XF(v) = xf {
                            row.copy_from_slice(&v);
                        }
                    });
                status.lev += stp;
                if status.lev >= 1 << haf_log {
                    status.phase = FftPhase::SufTranspose;
                    status.lev = 0;
                }
            },
        }
    }
    return (false, 0);
}


/// a step of sbs FFT for g1
pub fn sqrt_fft_fw_g1<E: Pairing>(
    sub_domain: &GeneralEvaluationDomain<<E as Pairing>::ScalarField>,
    sqrt_fft_info: &CustomParaFFT<E>,
    xg1: &mut Vec<E::G1>,
    status: &mut SqrtFftStatus,
    mem_for_transpose: &mut Vec<E::G1>,
    steps: usize
) -> (bool, usize) {
    let is_rev = status.is_rev;
    let haf_log = sqrt_fft_info.log_n >> 1;
    let mx_step: usize = 1<<haf_log;
    let mut steps_rem = steps;
    while steps_rem >0{
        match status.phase {
            FftPhase::PreTranspose | FftPhase::SufTranspose =>{
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_COPY);
                if status.lev < (1<< haf_log){
                    let cnt_stp = std::cmp::min(steps_rem, (mx_step -status.lev) /unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = status.lev  << haf_log;
                    let r = (status.lev+ stp) << haf_log;
                    (l..r).into_par_iter().zip(mem_for_transpose[l..r].par_iter_mut()).for_each(|(i,v)|{
                        let r = i >> haf_log;
                        let c = i&((1<<haf_log)-1);
                        *v = xg1[(c << haf_log) | r];
                    });
                    status.lev += stp;
                }
                else{
                    let cnt_stp = std::cmp::min(steps_rem, ((mx_step<<1) -status.lev) /unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = (status.lev-(1<<haf_log))  << haf_log;
                    let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                    xg1[l..r].copy_from_slice(&mem_for_transpose[l..r]);
                    status.lev += stp;
                }
                if status.lev >= 2<<haf_log{
                    if status.phase == FftPhase::PreTranspose{
                        status.lev = 0;
                        status.phase = FftPhase::ColFFT;
                    }
                    else{
                        status.lev = 0;
                        status.phase = FftPhase::PreTranspose;
                        return (true, steps_rem);
                    }
                }
            }
            FftPhase::ColFFT => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_FFT);
                let cnt_stp = std::cmp::min(steps_rem, (mx_step - status.lev)/unit_stp);
                let stp = cnt_stp * unit_stp;
                steps_rem -= cnt_stp;
                let haf_info = CustomParaFFT::<E>::new(1 << haf_log, sub_domain.group_gen());
                xg1[(status.lev << haf_log)..((status.lev + stp) << haf_log)]
                    .par_chunks_mut(1 << haf_log)
                    .for_each(|row| {
                        let mut xf = FFTVectorTypes::XG1(row.to_vec());
                        sqrt_fft(&haf_info, &mut xf, is_rev);
                        if let FFTVectorTypes::XG1(v) = xf {
                            row.copy_from_slice(&v);
                        }
                    });
                status.lev += stp;
                if status.lev >= 1 << haf_log {
                    status.phase = FftPhase::TRANSPOSE;
                    status.lev = 0;
                }
            },
            FftPhase::TRANSPOSE => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_TRANSPOSE);
                if status.lev < (1<< haf_log){
                    let cnt_stp = std::cmp::min(steps_rem, (mx_step-status.lev)/unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = status.lev  << haf_log;
                    let r = (status.lev+ stp) << haf_log;
                    (l..r).into_par_iter().zip(mem_for_transpose[l..r].par_iter_mut()).for_each(|(i,v)|{
                        let r = i >> haf_log;
                        let c = i&((1<<haf_log)-1);
                        *v = xg1[(c << haf_log) | r] * 
                            if status.is_rev {
                                sqrt_fft_info.omega_invs[r * c]
                            } else {
                                sqrt_fft_info.omegas[r * c]
                            }
                    });
                    status.lev += stp;
                }
                else{
                    let cnt_stp = std::cmp::min(steps_rem, ((mx_step<<1)-status.lev)/unit_stp);
                    let stp = cnt_stp * unit_stp;
                    steps_rem -= cnt_stp;
                    let l = (status.lev-(1<<haf_log))  << haf_log;
                    let r = (status.lev+stp-(1<<haf_log))  << haf_log;
                    xg1[l..r].copy_from_slice(&mem_for_transpose[l..r]);
                    status.lev += stp;
                    if status.lev >= 2 << haf_log {
                        status.phase = FftPhase::RowFFT;
                        status.lev = 0;
                    }
                }
            },
            FftPhase::RowFFT => {
                let unit_stp = std::cmp::min(mx_step, STEP_SQRT_FFT);
                let cnt_stp = std::cmp::min(steps_rem, (mx_step-status.lev)/unit_stp);
                let stp = cnt_stp * unit_stp;
                steps_rem -= cnt_stp;
                let haf_info = CustomParaFFT::<E>::new(1 << haf_log, sub_domain.group_gen());
                xg1[(status.lev << haf_log)..((status.lev + stp) << haf_log)]
                    .par_chunks_mut(1 << haf_log)
                    .for_each(|row| {
                        let mut xf = FFTVectorTypes::XG1(row.to_vec());
                        sqrt_fft(&haf_info, &mut xf, is_rev);
                        if let FFTVectorTypes::XG1(v) = xf {
                            row.copy_from_slice(&v);
                        }
                    });
                status.lev += stp;
                if status.lev >= 1 << haf_log {
                    status.phase = FftPhase::SufTranspose;
                    status.lev = 0;
                }
            },
        }
    }
    return (false, 0);
}

