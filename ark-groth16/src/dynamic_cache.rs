use crate::{Proof,Groth16,};
use crate::fft_handler::{
    sqrt_fft_fw_field, sqrt_fft_fw_g1, CustomParaFFT, SqrtFftStatus
};
use crate::data_structures::UpdatingKey;
use ark_ec::pairing::Pairing;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{ConstraintMatrices,SynthesisError};
use ark_std::vec::Vec;
use rayon::{prelude::*, join};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{Zero,UniformRand};
use crate::utils::{improved_lagrange_sparse_get,lagrange_of_dense_vec};
#[derive(Clone, Debug, PartialEq)]
/// Record the from field coeff for lagrange and p_i para
pub struct SparseRecord<E:Pairing>{
    /// Current Timestamp
    cur_tim: usize,
    /// Iteration for each slot
    tims: Vec<usize>,
    /// Value maintained
    vals: Vec<E::ScalarField>,
    /// useful indices
    u_ids: Vec<usize>,
}

impl<E:Pairing> SparseRecord<E>{
    /// Define a new SparseRecord
    pub fn new(n: usize) -> Self{
        let cur_tim: usize = 1;
        let tims: Vec<usize> = vec![0; n];
        let vals: Vec<E::ScalarField> = vec![E::ScalarField::zero(); n];
        let u_ids: Vec<usize> = Vec::with_capacity(n);
        Self { cur_tim, tims, vals, u_ids }
    }
    /// Start a new iterate version
    pub fn clr(&mut self){
        self.cur_tim += 1;
        self.u_ids.clear();
    }
    /// Update on specific slot
    pub fn ins(&mut self, id: usize, vl: E::ScalarField){
        if self.tims[id] != self.cur_tim{
            self.tims[id] = self.cur_tim;
            self.vals[id] = vl;
            self.u_ids.push(id);
        }
        else{
            self.vals[id] += vl;
        }
    }
    /// Check the i the bit
    pub fn get(&self, id: usize) -> E::ScalarField{
        if self.tims[id] == self.cur_tim{
            return self.vals[id];
        }
        E::ScalarField::zero()
    }
    /// Parse all the ids
    pub fn parse(&mut self) -> Vec<(usize, E::ScalarField)>{
        self.u_ids.sort();
        self.u_ids.par_iter().map(|id|{(*id, self.vals[*id])}).collect::<Vec<(usize, E::ScalarField)>>()
    }
    pub fn parse_unordered(&self) -> Vec<(usize, E::ScalarField)>{
        self.u_ids.par_iter().map(|id|{(*id, self.vals[*id])}).collect::<Vec<(usize, E::ScalarField)>>()
    }
}

///base batch size
pub const BASE_BATCH_SIZE: usize = 1<<10;
///Group operation batch size
pub const GROUP_OP_BATCH_SIZE: usize = BASE_BATCH_SIZE << 1;

/// Phase of Reconstruction
#[derive(Clone, Debug, PartialEq)]
pub enum ReconstructPhase{
    /// Prepare Vector Space
    VecPrepare(usize),
    /// DFT for Convolution 
    Dft(SqrtFftStatus),
    /// Pointwise Production
    PointwiseProduct(usize),
    /// iDFT for Convolution
    IDft(SqrtFftStatus),
    /// VectorIntegrate
    VecIntegrate(usize),
}


#[derive(Clone, Debug, PartialEq)]
/// Cache for all the intermediate info
pub struct FullCache<E: Pairing> {
    /// n, num of variable
    pub n: usize,
    /// domain
    pub haf_domain: GeneralEvaluationDomain<E::ScalarField>,
    /// accumulate A
    pub acc_a: Vec<E::ScalarField>,
    /// accumulate B
    pub acc_b: Vec<E::ScalarField>,
    /// G1; projection quotients of b_lagrange
    pub q_a: Vec<E::G1>,
    /// G1; projection quotients of b_lagrange
    pub q_b: Vec<E::G1>,
    /// G1: stores fl_a 
    pub fl_a: Vec<E::G1>,
    /// G1: stores fl_b 
    pub fl_b: Vec<E::G1>,
    /// G1: stores fw_a
    pub fw_a: Vec<E::ScalarField>,
    /// G1: stores fw_b
    pub fw_b: Vec<E::ScalarField>,
    /// extra mem fl_a
    pub mem_fl_a: Vec<E::G1>,
    /// extra mem fl_b
    pub mem_fl_b: Vec<E::G1>,
    /// extra mem fw_a
    pub mem_fw_a: Vec<E::ScalarField>,
    /// extra mem fw_b
    pub mem_fw_b: Vec<E::ScalarField>,
    /// info for modification on A
    pub mod_a: SparseRecord<E>,
    /// info for modification on B
    pub mod_b: SparseRecord<E>,
    /// Current stage
    pub status: ReconstructPhase,
    /// FFT parameter
    pub fft_info: CustomParaFFT<E>,
    /// half FFT parameter
    pub haf_fft_info: CustomParaFFT<E>,
}

/// Reconstruction Phase

impl<E:Pairing> FullCache<E>{
    ///new Full Cache
    pub fn new(
        n: usize, 
        acc_a: Vec<E::ScalarField>,
        acc_b: Vec<E::ScalarField>,
        q_a: Vec<E::G1>,
        q_b: Vec<E::G1>
    ) -> Self{
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        let haf_n: usize = 1 << (n.trailing_zeros()>>1);
        let haf_domain = GeneralEvaluationDomain::<E::ScalarField>::new(haf_n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        // let q_a: Vec<<E as Pairing>::G1> = Vec::with_capacity(n);
        // let q_b: Vec<<E as Pairing>::G1> = Vec::with_capacity(n);
        let fl_a: Vec<<E as Pairing>::G1> = vec![E::G1::zero(); n];
        let fl_b: Vec<<E as Pairing>::G1> = vec![E::G1::zero(); n];
        let fw_a: Vec<<E as Pairing>::ScalarField> = vec![E::ScalarField::zero(); n];
        let fw_b: Vec<<E as Pairing>::ScalarField> = vec![E::ScalarField::zero(); n];
        let mod_a: SparseRecord<E> = SparseRecord::<E>::new(n);
        let mod_b: SparseRecord<E> = SparseRecord::<E>::new(n);
        
        Self { n, haf_domain, acc_a, acc_b, q_a, q_b, fl_a, fl_b, fw_a, fw_b,
            mem_fl_a: vec![E::G1::zero(); n],
            mem_fl_b: vec![E::G1::zero(); n],
            mem_fw_a: vec![E::ScalarField::zero(); n],
            mem_fw_b: vec![E::ScalarField::zero(); n],
            mod_a, mod_b,
            status:ReconstructPhase::VecPrepare(0),
            fft_info: CustomParaFFT::<E>::new(n, domain.group_gen()),
            haf_fft_info: CustomParaFFT::<E>::new(haf_n, haf_domain.group_gen())
        }
    }
    /// begin the amortizing reconstruct
    pub fn flash(&mut self){
        self.status = ReconstructPhase::VecPrepare(0);
        self.mod_a.parse().into_iter().for_each(|(i,vl)| {self.acc_a[i] += vl;});
        self.mod_b.parse().into_iter().for_each(|(i,vl)| {self.acc_b[i] += vl;});
        self.mod_a.clr();
        self.mod_b.clr();
    }
    /// projective term between existing a and new delta b
    pub fn calc_update_a(&self, delta_a: &Vec<(usize, E::ScalarField)>) -> E::G1{
        delta_a.par_iter().map(|(i,a_i)|{
            self.q_b[*i] * a_i
        }).sum::<E::G1>()
    }
    /// projective term between existing b and new delta a
    pub fn calc_update_b(&self, delta_b: &Vec<(usize, E::ScalarField)>) -> E::G1{
        delta_b.par_iter().map(|(i,b_i)|{
            self.q_a[*i] * b_i
        }).sum::<E::G1>()
    }
    /// Take a step forward
    pub fn step_forward(&mut self, up: & UpdatingKey<E>, try_steps: usize) -> (bool, usize){
        let mut rem_stp = try_steps;
        while rem_stp > 0{
            let unt_stp = std::cmp::min(self.n, GROUP_OP_BATCH_SIZE);
            match self.status{
                ReconstructPhase::VecPrepare(ref mut id)=>{
                    let cnt_stp = std::cmp::min(rem_stp, (self.n-*id)/unt_stp);
                    let mx_stp = cnt_stp * unt_stp;
                    let res_a = &mut self.q_a[*id..(*id+mx_stp)];
                    let res_b = &mut self.q_b[*id..(*id+mx_stp)];
                    let f_a_slice = &self.acc_a[*id..(*id+mx_stp)];
                    let f_b_slice = &self.acc_b[*id..(*id+mx_stp)];
                    let fl_a_slice = &mut self.fl_a[*id..(*id+mx_stp)];
                    let fl_b_slice = &mut self.fl_b[*id..(*id+mx_stp)];
                    let fw_a_slice = &mut self.fw_a[*id..(*id+mx_stp)];
                    let fw_b_slice = &mut self.fw_b[*id..(*id+mx_stp)];
                    let lagrange = &up.lagrange_truncated[*id..(*id+mx_stp)];
                    let p = &up.p[*id..(*id+mx_stp)];
                    f_a_slice.par_iter().zip(f_b_slice.par_iter()).zip(
                    res_a.par_iter_mut().zip(res_b.par_iter_mut()).zip(
                        lagrange.par_iter().zip(p.par_iter()).zip(
                            fl_a_slice.par_iter_mut().zip(fl_b_slice.par_iter_mut())
                                .zip(fw_a_slice.par_iter_mut().zip(fw_b_slice.par_iter_mut()))
                            )
                        )
                    ).for_each(|((f_a,f_b),((qa,qb),((li, pi),((fl_a,fl_b),(fw_a,fw_b)))))|{
                        *fl_a = *li * f_a;
                        *fl_b = *li * f_b;
                        *fw_a = *f_a;
                        *fw_b = *f_b;
                        *qa = *pi * f_a;
                        *qb = *pi * f_b;
                    });
                    *id += mx_stp;
                    rem_stp -= cnt_stp;
                    if *id >= self.n{
                        *id = 0;
                        self.status = ReconstructPhase::Dft(SqrtFftStatus::new(false));
                    }
                }
                ReconstructPhase::Dft(ref mut fft_status)=>{
                    let mut fft_status_1a = fft_status.clone();
                    let mut fft_status_1b = fft_status.clone();
                    let mut fft_status_2a = fft_status.clone();
                    let mut fft_status_2b = fft_status.clone();
                    let (mut finish1a, mut finish1b, mut finish2a, mut finish2b) = ((false, 0), (false, 0), (false, 0), (false, 0));
                    {
                        let (mut mem_fl_a, mut mem_fl_b, mut mem_fw_a, mut mem_fw_b) = (&mut self.mem_fl_a, &mut self.mem_fl_b, &mut self.mem_fw_a, &mut self.mem_fw_b);
                        let (mut fl_a, mut fl_b, mut fw_a, mut fw_b) = (&mut self.fl_a, &mut self.fl_b, &mut self.fw_a, &mut self.fw_b);
                        join(
                            || {join(
                                || {finish1a = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut fl_a,&mut fft_status_1a, &mut mem_fl_a, rem_stp)},
                                || {finish1b = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut fl_b,&mut fft_status_1b, &mut mem_fl_b, rem_stp)},
                            )},
                            || {join(
                                || {finish2a = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut fw_a,&mut fft_status_2a, &mut mem_fw_a, rem_stp)},
                                || {finish2b = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut fw_b,&mut fft_status_2b, &mut mem_fw_b, rem_stp)},
                            )},
                        );
                    }
                    if finish1a != finish1b || finish1b != finish2a || finish2a !=finish2b{
                        println!("{:?}   {:?}   {:?}   {:?}", finish1a,finish1b,finish2a,finish2b);
                        println!("   {:?}\n   {:?}\n   {:?}\n   {:?}", fft_status_1a,fft_status_1b,fft_status_2a,fft_status_2b);
                        panic!("Not Syncronized");
                    }
                    rem_stp = finish1a.1;
                    if finish1a.0{
                        self.status = ReconstructPhase::PointwiseProduct(0);
                    }
                    else{
                        *fft_status = fft_status_1a;
                    }
                }
                ReconstructPhase::PointwiseProduct(ref mut id)=>{
                    let cnt_stp = std::cmp::min(rem_stp, (self.n-*id)/unt_stp);
                    let mx_stp = cnt_stp * unt_stp;
                    let fl_a_slice = &mut self.fl_a[*id..(*id+mx_stp)];
                    let fl_b_slice = &mut self.fl_b[*id..(*id+mx_stp)];
                    let fw_a_slice = &mut self.fw_a[*id..(*id+mx_stp)];
                    let fw_b_slice = &mut self.fw_b[*id..(*id+mx_stp)];
                    let w_slice = &up.w_for_q[*id..(*id+mx_stp)];
                    let w_rev_slice = &up.w_reverse_for_q[*id..(*id+mx_stp)];
                    w_slice.par_iter().zip(w_rev_slice.par_iter()).zip(
                    fl_a_slice.par_iter_mut().zip(fl_b_slice.par_iter_mut()).zip(
                        fw_a_slice.par_iter_mut().zip(fw_b_slice.par_iter_mut())
                        )
                    ).for_each(|((w,w_rev),((fl_a,fl_b),(fw_a,fw_b)))|{
                        *fl_a *= *w;
                        *fl_b *= *w;
                        *fw_a *= *w_rev;
                        *fw_b *= *w_rev;
                    });
                    *id += mx_stp;
                    rem_stp -= cnt_stp;
                    if *id >= self.n{
                        *id = 0;
                        self.status = ReconstructPhase::IDft(SqrtFftStatus::new(true));
                    }
                }
                ReconstructPhase::IDft(ref mut fft_status)=>{
                    let mut fft_status_1a = fft_status.clone();
                    let mut fft_status_1b = fft_status.clone();
                    let mut fft_status_2a = fft_status.clone();
                    let mut fft_status_2b = fft_status.clone();
                    let (mut finish1a, mut finish1b, mut finish2a, mut finish2b) = ((false, 0), (false, 0), (false, 0), (false, 0));
                    {
                        let (mut mem_fl_a, mut mem_fl_b, mut mem_fw_a, mut mem_fw_b) = (&mut self.mem_fl_a, &mut self.mem_fl_b, &mut self.mem_fw_a, &mut self.mem_fw_b);
                        let (mut fl_a, mut fl_b, mut fw_a, mut fw_b) = (&mut self.fl_a, &mut self.fl_b, &mut self.fw_a, &mut self.fw_b);
                        join(
                            || {join(
                                || {finish1a = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut fl_a,&mut fft_status_1a, &mut mem_fl_a, rem_stp)},
                                || {finish1b = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut fl_b,&mut fft_status_1b, &mut mem_fl_b, rem_stp)},
                            )},|| {join(
                                || {finish2a = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut fw_a,&mut fft_status_2a, &mut mem_fw_a, rem_stp)},
                                || {finish2b = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut fw_b,&mut fft_status_2b, &mut mem_fw_b, rem_stp)},
                            )},
                        );
                    }
                    // let finish1a = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut self.fl_a,&mut fft_status_1a, &mut self.mem_fl_a);
                    // let finish1b = sqrt_fft_fw_g1(&self.haf_domain,&self.fft_info,&mut self.fl_b,&mut fft_status_1b, &mut self.mem_fl_b);
                    // let finish2a = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut self.fw_a,&mut fft_status_2a, &mut self.mem_fw_a);
                    // let finish2b = sqrt_fft_fw_field(&self.haf_domain,&self.fft_info,&mut self.fw_b,&mut fft_status_2b, &mut self.mem_fw_b);
                    if finish1a != finish1b || finish1b != finish2a || finish2a !=finish2b{
                        panic!("Not Syncronized");
                    }
                    rem_stp = finish1a.1;
                    if finish1a.0{
                        self.status = ReconstructPhase::VecIntegrate(0);
                    }
                    else{
                        *fft_status = fft_status_1a;
                    }
                }
                ReconstructPhase::VecIntegrate(ref mut id)=>{
                    let cnt_stp = std::cmp::min(rem_stp, (self.n-*id)/unt_stp);
                    let mx_stp = cnt_stp * unt_stp;
                    let res_a = &mut self.q_a[*id..(*id+mx_stp)];
                    let res_b = &mut self.q_b[*id..(*id+mx_stp)];
                    let fl_a_slice = &mut self.fl_a[*id..(*id+mx_stp)];
                    let fl_b_slice = &mut self.fl_b[*id..(*id+mx_stp)];
                    let fw_a_slice = &mut self.fw_a[*id..(*id+mx_stp)];
                    let fw_b_slice = &mut self.fw_b[*id..(*id+mx_stp)];
                    let lagrange = &up.lagrange_truncated[*id..(*id+mx_stp)];
                    res_a.par_iter_mut().zip(res_b.par_iter_mut()).zip(
                    lagrange.par_iter().zip(
                        fl_a_slice.par_iter_mut().zip(fl_b_slice.par_iter_mut())
                            .zip(fw_a_slice.par_iter_mut().zip(fw_b_slice.par_iter_mut()))
                        )
                    ).for_each(|((qa,qb),(li,((fl_a,fl_b),(fw_a,fw_b))))|{
                        *qa += *fl_a + *li * fw_a;
                        *qb += *fl_b + *li * fw_b;
                    });
                    *id += mx_stp;
                    rem_stp -= cnt_stp;
                    if *id >= self.n{
                        *id = 0;
                        self.status = ReconstructPhase::VecPrepare(0);
                        return (true, rem_stp);
                    }
                }
            }
        };
        (false, rem_stp)
    }
}

/// checkpoints structure
#[derive(Clone, Debug, PartialEq)]
pub struct FullyCheckpoint<E:Pairing>{
    /// Vector length
    n: usize,
    /// Cache
    pub cache: FullCache<E>,
    // /// delta on A
    // delta_a: SparseRecord<E>,
    // /// delta on B
    // delta_b: SparseRecord<E>,
    /// The `A` element in `G1`.
    pub raw_a_g1: E::G1,
    /// The `B` element in `G2`.
    pub raw_b_g1: E::G1,
    /// The `B` element in `G2`.
    pub raw_b_g2: E::G2,
    /// The `C` element in `G1`.
    pub raw_c_g1: E::G1,
    // /// Static vector
    // lag_coef_cache: SparseRecord<E>
}
type D<F> = GeneralEvaluationDomain<F>;
impl<E:Pairing> FullyCheckpoint<E>{
    /// Initialize Checkpoint
    pub fn new(n: usize,
        uk: &UpdatingKey<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
    ) -> Self{
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        let (acc_a, acc_b, c) = lagrange_of_dense_vec(&r1cs, &[instance, witness].concat()).unwrap();
        let (q_a, q_b) = Groth16::<E>::compute_projection_quotient(uk, domain, &acc_a.clone(), &acc_b.clone(), &c);
        let (a_g1,b_g1,b_g2,c_g1) =
        Groth16::<E>::calc_raw_proof(&uk.pk, &domain, &instance, &witness, &acc_a, &acc_b, &c);

        Self{
            n,
            cache:    FullCache::<E>::new(n, acc_a, acc_b, q_a, q_b),
            // delta_a:  SparseRecord::<E>::new(n),
            // delta_b:  SparseRecord::<E>::new(n), 
            raw_a_g1: a_g1,
            raw_b_g1: b_g1,
            raw_b_g2: b_g2,
            raw_c_g1: c_g1,
            // lag_coef_cache: SparseRecord::<E>::new(n),
        }
    }
    ///new with given cached quotient
    pub fn new_with_qached_quotient(n: usize,
        uk: &UpdatingKey<E>,
        r1cs: &ConstraintMatrices<E::ScalarField>,
        instance: &[E::ScalarField],
        witness: &[E::ScalarField],
        qa: &Vec<E::G1>,
        qb: &Vec<E::G1>, 
    ) -> Self{
        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(n)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)
            .unwrap();
        let (acc_a, acc_b, c) = lagrange_of_dense_vec(&r1cs, &[instance, witness].concat()).unwrap();
        // let (q_a, q_b) = Groth16::<E>::compute_projection_quotient(uk, domain, &acc_a.clone(), &acc_b.clone(), &c);
        let (a_g1,b_g1,b_g2,c_g1) =
        Groth16::<E>::calc_raw_proof(&uk.pk, &domain, &instance, &witness, &acc_a, &acc_b, &c);

        Self{
            n,
            cache:    FullCache::<E>::new(n, acc_a, acc_b, qa.clone(), qb.clone()),
            // delta_a:  SparseRecord::<E>::new(n),
            // delta_b:  SparseRecord::<E>::new(n), 
            raw_a_g1: a_g1,
            raw_b_g1: b_g1,
            raw_b_g2: b_g2,
            raw_c_g1: c_g1,
            // lag_coef_cache: SparseRecord::<E>::new(n),
        }
    }
    /// increment A in a_log
    pub fn increment_a(&mut self, a_lag: &Vec<(usize, E::ScalarField)>){
        a_lag.iter().for_each(|(i,vl)| {
            self.cache.mod_a.ins(*i, *vl);
        });
    }
    /// increment B in b_log
    pub fn increment_b(&mut self, b_lag: &Vec<(usize, E::ScalarField)>){
        b_lag.iter().for_each(|(i,vl)| {
            self.cache.mod_b.ins(*i,*vl);
        });
    }
    ///Initialize to begin reconstruction
    pub fn prepare_for_reconstruct(&mut self){
        self.cache.flash();
    }
    ///Move a step forward
    pub fn mv_forward(&mut self, uk:&UpdatingKey<E>, steps: usize) -> (bool, usize){
        self.cache.step_forward(&uk, steps)
    }
    /// d^2 compute the cross term contribution
    pub fn calc_cross_terms(
        &self,
        uk: &UpdatingKey<E>,
        a_lagrange_sparse: &Vec<(usize, E::ScalarField)>,
        b_lagrange_sparse: &Vec<(usize, E::ScalarField)>
    )-> E::G1{
        // println!("Crossing:");
        // println!("   {:?}", a_lagrange_sparse);
        // println!("   {:?}", b_lagrange_sparse);
        // println!(" ===================\n ");
        let c_cross_1 = a_lagrange_sparse.par_iter().map(|(i,a)|{
            let ot_sum = b_lagrange_sparse.par_iter().map(|(j,b)|{
                if i==j {E::ScalarField::zero()} else {uk.w[(self.n + j - i) % self.n]*b} 
            }).sum::<E::ScalarField>();
            uk.lagrange_truncated[*i] * (*a * ot_sum)
        }).sum::<E::G1>();

        let c_cross_2 = b_lagrange_sparse.par_iter().map(|(j,b)|{
            let ot_sum = a_lagrange_sparse.par_iter().map(|(i,a)|{
                if i==j {E::ScalarField::zero()} else {uk.w[(self.n + *i - *j) % self.n]*a} 
            }).sum::<E::ScalarField>();
            uk.lagrange_truncated[*j] * (*b * ot_sum)
        }).sum::<E::G1>();

        
        let mut results = Vec::<(usize, E::ScalarField)>::new();
        let mut cur_i = 0;
        let mut cur_j = 0;
        let mut a_copy = a_lagrange_sparse.clone();
        let mut b_copy = b_lagrange_sparse.clone();
        a_copy.sort_by_key(|x| x.0);
        b_copy.sort_by_key(|y| y.0);
        // two-pointer merge join
        while cur_i < a_copy.len() && cur_j < b_copy.len() {
            match a_copy[cur_i].0.cmp(&b_copy[cur_j].0) {
                std::cmp::Ordering::Less => cur_i += 1,
                std::cmp::Ordering::Greater => cur_j += 1,
                std::cmp::Ordering::Equal => {
                    results.push((a_copy[cur_i].0, a_copy[cur_i].1.clone() * b_copy[cur_j].1.clone()));
                    cur_i += 1;
                    cur_j += 1;
                }
            }
        }
        let c_cross_3 = results.par_iter().map(|(i,val)| uk.p[*i] * val).sum::<E::G1>();
    return c_cross_1 + c_cross_2 + c_cross_3
}

    /// Compute Semi Proof
    pub fn update_from(&mut self, uk: &UpdatingKey<E>,
        // matrices: &ConstraintMatrices<E::ScalarField>,
        num_instance: usize,
        num_witness: usize,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
        ){
        let update = [&instance_update[..], &witness_update[..]].concat();
        // println!("Update is {:?}",update);
        let a_g1_new: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.u_g1[*i].into_group() * update_i)
            .sum();

        let b_g1_new: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.v_g1[*i].into_group() * update_i)
            .sum();

        let b_g2_new: E::G2 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.v_g2[*i].into_group() * update_i)
            .sum();

        let (a_lagrange_sparse, b_lagrange_sparse) =
            improved_lagrange_sparse_get(&uk.a_matrix_cols, &uk.b_matrix_cols, &update).unwrap();

        // println!("Witness_Update: {:?}", witness_update.iter().map(|(i, vl)|*i).collect::<Vec<usize>>());
        // println!("Cross A({:?}) B({:?})", a_lagrange_sparse.len(), b_lagrange_sparse.len());
        // println!("A_lag: {:?}", a_lagrange_sparse.iter().map(|(i, vl)|*i).collect::<Vec<usize>>());
        // println!("B_lag: {:?}", b_lagrange_sparse.iter().map(|(i, vl)|*i).collect::<Vec<usize>>());
        let c_project = self.cache.calc_update_a(&a_lagrange_sparse) + self.cache.calc_update_b(&b_lagrange_sparse);

        let update_c_g1_linear_combination: E::G1 = witness_update
            .par_iter()
            .map(|(i, update_i)| uk.pk.t_p_g1[*i - num_instance].into_group() * update_i)
            .sum::<E::G1>();
        
        // a_lagrange_sparse.iter().for_each(|(i,vl)|{self.cache.ins(*i, *vl);});
        self.increment_a(&a_lagrange_sparse);
        let new_a_parse = self.cache.mod_a.parse();
        let old_b_parse = self.cache.mod_b.parse();
        self.increment_b(&b_lagrange_sparse);
        // b_lagrange_sparse.iter().for_each(|(i,vl)|{self.delta_b.ins(*i, *vl);});
 
        let c_new = update_c_g1_linear_combination + c_project
            + self.calc_cross_terms(&uk, &new_a_parse, &b_lagrange_sparse)
            + self.calc_cross_terms(&uk, &a_lagrange_sparse, &old_b_parse);

        self.raw_a_g1 += a_g1_new;
        self.raw_b_g1 += b_g1_new;
        self.raw_b_g2 += b_g2_new;
        self.raw_c_g1 += c_new;
    }

    /// simulate a update without changing it self
    pub fn sim_update_from(&self, uk: &UpdatingKey<E>,
        num_instance: usize,
        num_witness: usize,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)],
        )->(E::G1,E::G1,E::G2,E::G1){
        let update = [&instance_update[..], &witness_update[..]].concat();
        // println!("Update is {:?}",update);
        let a_g1_new: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.u_g1[*i].into_group() * update_i)
            .sum();

        let b_g1_new: E::G1 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.v_g1[*i].into_group() * update_i)
            .sum();

        let b_g2_new: E::G2 = update
            .par_iter()
            .map(|(i, update_i)| uk.pk.v_g2[*i].into_group() * update_i)
            .sum();

        let (a_lagrange_sparse, b_lagrange_sparse) =
            improved_lagrange_sparse_get(&uk.a_matrix_cols, &uk.b_matrix_cols, &update).unwrap();

        // println!("Witness_Update: {:?}", witness_update.iter().map(|(i, vl)|*i).collect::<Vec<usize>>());
        // println!("Cross A({:?}) B({:?})", a_lagrange_sparse.len(), b_lagrange_sparse.len());
        // println!("A_lag: {:?}", a_lagrange_sparse.iter().map(|(i, vl)|*i).collect::<Vec<usize>>());
        // println!("B_lag: {:?}", b_lagrange_sparse.iter().map(|(i, vl)|*i).collect::<Vec<usize>>());
        let c_project = self.cache.calc_update_a(&a_lagrange_sparse) + self.cache.calc_update_b(&b_lagrange_sparse);

        let update_c_g1_linear_combination: E::G1 = witness_update
            .par_iter()
            .map(|(i, update_i)| uk.pk.t_p_g1[*i - num_instance].into_group() * update_i)
            .sum::<E::G1>();
        
        // a_lagrange_sparse.iter().for_each(|(i,vl)|{self.cache.ins(*i, *vl);});
        // self.increment_a(&a_lagrange_sparse);
        let mut cur_a_sparse: Vec<(usize, E::ScalarField)> = self.cache.mod_a.parse_unordered();
        let mut new_a_sparse: Vec<(usize, E::ScalarField)> = Vec::new();
        cur_a_sparse.append(&mut a_lagrange_sparse.clone());
        cur_a_sparse.sort();
        cur_a_sparse.into_iter().for_each(|(i,vl)|{
            if let Some((last_id,last_vl)) = new_a_sparse.last_mut(){
                if *last_id == i{
                    *last_vl += vl;
                }
                else{
                    new_a_sparse.push((i,vl));
                }
            }
            else{
                new_a_sparse.push((i,vl));
            }
        });
        self.cache.mod_a.parse_unordered();
        let old_b_parse = self.cache.mod_b.parse_unordered();
        // self.increment_b(&b_lagrange_sparse);
        // b_lagrange_sparse.iter().for_each(|(i,vl)|{self.delta_b.ins(*i, *vl);});
 
        let c_new = update_c_g1_linear_combination + c_project
            + self.calc_cross_terms(&uk, &new_a_sparse, &b_lagrange_sparse)
            + self.calc_cross_terms(&uk, &a_lagrange_sparse, &old_b_parse);

        (self.raw_a_g1 + a_g1_new,
        self.raw_b_g1 + b_g1_new,
        self.raw_b_g2 + b_g2_new,
        self.raw_c_g1 + c_new)
    }

    /// Trivially append the delta vec to the tail
    pub fn trivial_append(&mut self,
        uk: &UpdatingKey<E>,
        instance_update: &[(usize, E::ScalarField)],
        witness_update: &[(usize, E::ScalarField)]
    ){
        let update = [&instance_update[..], &witness_update[..]].concat();
        let (a_lagrange_sparse, b_lagrange_sparse) =
            improved_lagrange_sparse_get(&uk.a_matrix_cols, &uk.b_matrix_cols, &update).unwrap();
        self.increment_a(&a_lagrange_sparse);
        self.increment_b(&b_lagrange_sparse);
    }
    /// Compute Final Proof
    pub fn get_proof_with_rs(&mut self, uk: &UpdatingKey<E>,
    ) -> Proof<E>{
        // self.update_from(uk, matrices, instance_update, witness_update);
        let r: <E as Pairing>::ScalarField = E::ScalarField::rand(&mut ark_std::test_rng());
        let s: <E as Pairing>::ScalarField = E::ScalarField::rand(&mut ark_std::test_rng());
        // let r: <E as Pairing>::ScalarField = E::ScalarField::zero();
        // let s: <E as Pairing>::ScalarField = E::ScalarField::zero();
        let a_g1 = self.raw_a_g1 + uk.pk.delta_g1 * r;
        let b_g1 = self.raw_b_g1 + uk.pk.delta_g1 * s;
        let b_g2 = self.raw_b_g2 + uk.pk.vk.delta_g2.into_group() * s;
        let c_g1 = self.raw_c_g1 + a_g1 * s + b_g1 * r - uk.pk.delta_g1*(r * s) ;
        Proof { a_g1:a_g1.into_affine(), b_g2: b_g2.into_affine(), c_g1: c_g1.into_affine()} 
    }
    /// Copy current raw proof
    pub fn copy_proof_from(&mut self, other: &FullyCheckpoint<E>){
        self.raw_a_g1 = other.raw_a_g1;
        self.raw_b_g1 = other.raw_b_g1;
        self.raw_b_g2 = other.raw_b_g2;
        self.raw_c_g1 = other.raw_c_g1;
    }
}
