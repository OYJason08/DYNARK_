use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_groth16::{prepare_verifying_key, Groth16, Proof, UpdatingKey};
use ark_ff::UniformRand;
use ark_ec::{AffineRepr, CurveGroup};
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng,
};
use std::{
    fs::File,
    io::Write,
    time::Instant,
};
use ark_groth16::instance_generator::{generate_update_once, generate_matrices, generate_instance_witness};
use ark_groth16::dynamic_cache::{FullyCheckpoint};
use ark_groth16::instance_generator::{generate_update};
use std::time::Duration;
use rayon::ThreadPoolBuilder;
//  RAYON_NUM_THREADS=1 cargo run --example bench_v2 --no-default-features --features "std parallel" --release
//  cargo run --example bench_v2 --release

fn main() {
    // Test_Preprocess_STD::<Bls12_381>(5);
    // Test_ProveTime_Semi::<Bls12_381>(22,50, true);
    // Test_ProveTime_Semi::<Bls12_381>(20,10, false);
    // Test_ProveTime_Fully::<Bls12_381>(22, 20, 32);
    Test_EveryThing_On_N::<Bls12_381>(12, false, 3, 31);
}

fn Test_EveryThing_On_N<E: Pairing>(log_n: usize, para: bool, repeat_time: usize, non_para_every_k: usize) {
    let mut file= match File::create(format!("Para({:?})_EveryThing_On_N=(2^{})_Bench.txt",para,log_n)){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    let num_update = 1;
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    let num_assignment = 1 << log_n;
    let num_instance = 1;
    let num_witness = num_assignment - num_instance;
    let num_constraint = num_assignment;

    let (matrices, ka, kb, ca, cb) =
        generate_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
    let (mut instance, mut witness) = generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);
    let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynarc(matrices.clone(), &mut rng).unwrap();
    
    let (_, mut cache) =
        Groth16::<E>::prove_dynarc(&pk, &matrices, &instance, &witness, &mut rng).unwrap();

    let pvk = prepare_verifying_key(&vk);
    let uk = Groth16::<E>::generate_updating_keys(&matrices, &domain, &pk).unwrap();
    println!("                              Got Update Key");
    Groth16::<E>::process_dynarc(&uk, &matrices, &instance, &witness, &mut cache).unwrap();
    println!("                              Got Cached Quotient");
    let mut checkpoint_using: FullyCheckpoint<E> = FullyCheckpoint::<E>::new_with_qached_quotient(
        num_constraint, &uk, &matrices, &instance, &witness, &cache.q_a, &cache.q_b);
    let mut checkpoint_preparing =  checkpoint_using.clone();
// Groth16
    let prover_time = (0..repeat_time).into_iter().map(|_|{
        let mut prove_time = Duration::ZERO;
        let mut proof = Proof::<E>::default();
        let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(if para{192} else {1}).build().unwrap();

        pool.install(||{
            let prove_start = Instant::now();
            (proof, _) =
                Groth16::<E>::prove_dynarc(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
            prove_time = prove_start.elapsed();
            // writeln!(file, "Standard Groth16 Prove Size: {:?}\n", proof_size);
            
        });
        let result = Groth16::<E>::verify_dynarc(&pvk, &proof, &instance).unwrap();
        if !result{
            panic!("Prove Fail");
        }
        prove_time
    }).sum::<Duration>() / repeat_time.try_into().unwrap();
    println!("Groth16 Prove time: {:?}", prover_time);
    writeln!(file, " Groth16 Prove time: {:?}", prover_time);
// Semi
    let prove_time = (0..(repeat_time<<5)).into_iter().map(|_|{
        let prove_time = Duration::ZERO;
        let (instance_update, witness_update) =
            generate_update_once::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &instance, &witness, &mut rng);
        let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(if para{192} else {1}).build().unwrap();
        let mut proof_updated =  Proof::<E>::default();
        let mut semi_prover_update_time = Duration::default();
        pool.install(||{
            let prove_start = Instant::now();
            proof_updated = Groth16::<E>::update_dynarc(&uk, &matrices, &instance_update, &witness_update, &cache).unwrap();
            semi_prover_update_time = prove_start.elapsed();
        });
        let result = Groth16::<E>::verify_dynarc(&pvk, &proof_updated, &instance).unwrap();
        if !result{
            panic!("Semi Prover Fail");
        }
        semi_prover_update_time
    }).sum::<Duration>() / (repeat_time<<5).try_into().unwrap();
    println!("Semi Prove time: {:?}", prove_time);
    writeln!(file, "Semi Prove time: {:?}", prove_time);
// Fully
    let mut fully_prover_no_para_time: Duration = Duration::ZERO;
    let mut fully_prover_para_time: Duration = Duration::ZERO;
    let mut fully_prover_no_para_cnt:usize = 0;
    let mut fully_prover_para_cnt:usize = 0;
    let repeat_time = ((1<<((log_n+6)>>1))/num_update)/((non_para_every_k<<1)+1)+1;
    (0..repeat_time).into_iter().for_each(|_|{
        (0..((non_para_every_k<<1)+1)).into_iter().for_each(|it|{

            let (instance_update, witness_update) =
                generate_update::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng);
            let mut rem_stp = num_update;            
            let cond = it != non_para_every_k;
            let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(if cond {192} else {1}).build().unwrap();
            let mut proof_updated =  Proof::<E>::default();
            let mut fully_prover_update_time: Duration = Duration::ZERO;
            let mut start_new_checkpoint = false;
            pool.install(|| {
                let fully_prover_start = Instant::now();
                while rem_stp > 0{
                    let res = checkpoint_preparing.mv_forward(&uk, rem_stp);
                    rem_stp = res.1;
                    if res.0{
                        checkpoint_preparing.copy_proof_from(&checkpoint_using);
                        std::mem::swap(&mut checkpoint_preparing,&mut checkpoint_using);
                        checkpoint_preparing.prepare_for_reconstruct();
                        if start_new_checkpoint{
                            break;
                        }
                        start_new_checkpoint = true;
                    }
                }
                
                checkpoint_preparing.trivial_append(&uk, &instance_update, &witness_update);
                checkpoint_using.update_from(&uk, num_instance, num_witness,&instance_update, &witness_update);
                proof_updated = checkpoint_using.get_proof_with_rs(&uk);
                
                fully_prover_update_time = fully_prover_start.elapsed();
            });
            let result = Groth16::<E>::verify_dynarc(&pvk, &proof_updated, &instance).unwrap();
            if !result {
                panic!(" Fully prover Verifying Failed");
            }
            if cond{
                fully_prover_para_time += fully_prover_update_time;
                fully_prover_para_cnt += 1;
            }else{
                fully_prover_no_para_time += fully_prover_update_time;
                fully_prover_no_para_cnt += 1;
            }
        });
    });
    fully_prover_para_time /= fully_prover_para_cnt.try_into().unwrap();
    if fully_prover_no_para_cnt > 0{
        fully_prover_no_para_time /= fully_prover_no_para_cnt.try_into().unwrap();
    }
    println!("Fully dynamic Prover Avg Parallel Time: {:?}", fully_prover_para_time);
    writeln!(file, "Fully dynamic Prover Avg Parallel Time: {:?}", fully_prover_para_time);
    println!("Fully dynamic Prover Avg Non-Parallel Time: {:?}", fully_prover_no_para_time);
    writeln!(file, "Fully dynamic Prover Avg Non-Parallel Time: {:?}", fully_prover_no_para_time);
}

/// Test Preprocessing
fn Test_Preprocess_STD<E: Pairing>(repeat_time:usize) {
    let mut file= match File::create("NoPara_Preprocessing_Standard.txt"){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    (10..21).into_iter().for_each(|domain_log_n: usize|{
        let num_assignment = 1 << domain_log_n;
        let num_instance = 1;
        let num_witness = num_assignment - num_instance;
        let num_constraint = num_assignment;

        let (matrices, ka, kb, ca, cb) =
            generate_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
        let (instance, witness) = generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);
        let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynarc(matrices.clone(), &mut rng).unwrap();
        
        let pvk = prepare_verifying_key(&vk);

        println!("Start For N = {:?}\n", num_assignment);
        writeln!(file, "Start For N = {:?}\n", num_assignment);
        let groth16_processing_time = (0..repeat_time).into_iter().map(|_|{
            let prove_start = Instant::now();
            let (proof, cache) =
                Groth16::<E>::prove_dynarc(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
            let prove_time = prove_start.elapsed();
            let result = Groth16::<E>::verify_dynarc(&pvk, &proof, &instance).unwrap();
            println!("        Groth16 Prove time: {:?},    Result = {:?}", prove_time, result);
            if !result{
                panic!("Prove Fail");
            }
            // writeln!(file, "Standard Groth16 Prove Size: {:?}\n", proof_size);
            prove_time
        }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
        println!("Standard Groth16 Prove Avg Time: {:?}\n----------------------------------------------------------------", groth16_processing_time);
        writeln!(file, "Standard Groth16 Prove Avg Time: {:?}\n----------------------------------------------------------------", groth16_processing_time);
    });

}



/// Test Prover Time for Semi
fn Test_ProveTime_Semi<E: Pairing>(max_log_n:usize,repeat_time:usize, para: bool) {
    let mut file= match File::create(format!("Para({:?})_EverySemi_Until_N=(2^{}).txt",para,max_log_n)){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    for log_n in (10..(max_log_n+1)).step_by(2){
        let N: usize = 1<<log_n;
        let num_assignment = N;
        let num_instance = 1;
        let num_witness = num_assignment - num_instance;
        let num_constraint = num_assignment;

        let (matrices, ka, kb, ca, cb) =
            generate_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
        let (instance, witness) = generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);
        let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynarc(matrices.clone(), &mut rng).unwrap();
        println!("                              Got Prover key and Verifier Key");
        let pvk = prepare_verifying_key(&vk);
        let uk = Groth16::<E>::generate_updating_keys(&matrices, &domain, &pk).unwrap();
        println!("                              Got Update Key");
        let (_, mut cache) =
            Groth16::<E>::prove_dynarc(&pk, &matrices, &instance, &witness, &mut rng).unwrap();

        Groth16::<E>::process_dynarc(&uk, &matrices, &instance, &witness, &mut cache).unwrap();
        println!("                              Got Cached Quotient");

        let instance_copy = instance.clone();
        let witness_copy = witness.clone();

        // let mut checkpoint_using: FullyCheckpoint<E> = FullyCheckpoint::<E>::new_with_qached_quotient(
        //     num_constraint, &uk, &matrices, &instance, &witness, &cache.q_a, &cache.q_b);
        // let mut checkpoint_preparing =  checkpoint_using.clone();
        // println!("                              Got Fully Checkpoint");

        let dist_from_last_renew:usize = 0;

        let list_log_d = if log_n !=20 {vec![0usize]} else {(0usize..std::cmp::min(log_n-1,18usize)).collect::<Vec<usize>>()};
        // let list_log_d = (0..18).collect::<Vec<usize>>();
        list_log_d.iter().for_each(|log_d|{
            let num_update:usize = 1<<log_d;
            println!("Semi for d = {:?} with N = {:?}\n", num_update, num_assignment);
            writeln!(file, "Semi for d = {:?} with N = {:?}\n", num_update, num_assignment);
            
            let semi_prover_time = (0..repeat_time).into_iter().map(|_|{
                let (instance_update, witness_update) =
                    generate_update_once::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &instance_copy, &witness_copy, &mut rng);
                let semi_prover_start = Instant::now();

                let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(if para{192} else {1}).build().unwrap();
                let mut proof_updated =  Proof::<E>::default();
                let mut semi_prover_update_time = Duration::default();
                pool.install(|| {
                    let prove_start = Instant::now();
                    proof_updated = Groth16::<E>::update_dynarc(&uk, &matrices, &instance_update, &witness_update, &cache).unwrap();
                    semi_prover_update_time = prove_start.elapsed();
                });

                let result = Groth16::<E>::verify_dynarc(&pvk, &proof_updated, &instance_copy).unwrap();
                // println!("Dynarc New proof verification result: >>>> {} <<<<", result);
                if !result{
                    panic!("Semi Prover Fail");
                }
                semi_prover_update_time
            }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
            println!("Semi dynamic Prover Avg Time: {:?}\n", semi_prover_time);
            writeln!(file, "Semi dynamic Prover Avg Time: {:?}\n", semi_prover_time);
        });

        println!("================================================================");
        writeln!(file, "----------------------------------------------------------------");
    }
    
}



/// Test Prover Time for Semi
fn Test_ProveTime_Fully<E: Pairing>(max_log_n:usize, detail_log_n:usize, non_para_every_k:usize) {
    let mut file= match File::create(format!("Append_EveryFully_Until_N=(2^{}).txt", max_log_n)){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    for log_n in (22..(max_log_n+1)).step_by(2){
        let N: usize = 1<<log_n;
        let num_assignment = N;
        let num_instance = 1;
        let num_witness = num_assignment - num_instance;
        let num_constraint = num_assignment;

        let (matrices, ka, kb, ca, cb) =
            generate_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
        let (mut instance, mut witness) = generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);
        let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynarc(matrices.clone(), &mut rng).unwrap();
        println!("                              Got Prover key and Verifier Key");
        let pvk = prepare_verifying_key(&vk);
        let uk = Groth16::<E>::generate_updating_keys(&matrices, &domain, &pk).unwrap();
        println!("                              Got Update Key");
        let (_, mut cache) =
            Groth16::<E>::prove_dynarc(&pk, &matrices, &instance, &witness, &mut rng).unwrap();

        Groth16::<E>::process_dynarc(&uk, &matrices, &instance, &witness, &mut cache).unwrap();
        println!("                              Got Cached Quotient");


        let mut checkpoint_using: FullyCheckpoint<E> = FullyCheckpoint::<E>::new_with_qached_quotient(
            num_constraint, &uk, &matrices, &instance, &witness, &cache.q_a, &cache.q_b);
        let mut checkpoint_preparing =  checkpoint_using.clone();
        println!("                              Got Fully Checkpoint");


        let list_log_d = if log_n !=detail_log_n {vec![0usize]} else {(0usize..std::cmp::min(log_n-1,11usize)).collect::<Vec<usize>>()};
        // let list_log_d = (0..18).collect::<Vec<usize>>();

        list_log_d.iter().for_each(|log_d: &usize|{
            let num_update:usize = 1<<log_d;
            println!("Fully for d = {:?} with N = {:?}\n", num_update, num_assignment);
            writeln!(file, "Fully for d = {:?} with N = {:?}\n", num_update, num_assignment);
            let mut fully_prover_no_para_time: Duration = Duration::ZERO;
            let mut fully_prover_para_time: Duration = Duration::ZERO;
            let mut fully_prover_no_para_cnt:usize = 0;
            let mut fully_prover_para_cnt:usize = 0;
            let repeat_time = ((1<<((log_n+6)>>1))/num_update)/((non_para_every_k<<1)+1)+1;
            (0..repeat_time).into_iter().for_each(|_|{
                (0..((non_para_every_k<<1)+1)).into_iter().for_each(|it|{

                    let (instance_update, witness_update) =
                        generate_update::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng);
                    let mut rem_stp = num_update;            
                    let cond = it != non_para_every_k || log_n > 20;
                    let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(if cond {192} else {1}).build().unwrap();
                    let mut proof_updated =  Proof::<E>::default();
                    let mut fully_prover_update_time: Duration = Duration::ZERO;
                    let mut start_new_checkpoint = false;
                    pool.install(|| {
                        let fully_prover_start = Instant::now();
                        while rem_stp > 0{
                            let res = checkpoint_preparing.mv_forward(&uk, rem_stp);
                            rem_stp = res.1;
                            if res.0{
                                checkpoint_preparing.copy_proof_from(&checkpoint_using);
                                std::mem::swap(&mut checkpoint_preparing,&mut checkpoint_using);
                                checkpoint_preparing.prepare_for_reconstruct();
                                if start_new_checkpoint{
                                    break;
                                }
                                start_new_checkpoint = true;
                            }
                        }
                        
                        checkpoint_preparing.trivial_append(&uk, &instance_update, &witness_update);
                        checkpoint_using.update_from(&uk, num_instance, num_witness,&instance_update, &witness_update);
                        proof_updated = checkpoint_using.get_proof_with_rs(&uk);
                        
                        fully_prover_update_time = fully_prover_start.elapsed();
                    });
                    let result = Groth16::<E>::verify_dynarc(&pvk, &proof_updated, &instance).unwrap();
                    if !result {
                        panic!(" Fully prover Verifying Failed");
                    }
                    if cond{
                        fully_prover_para_time += fully_prover_update_time;
                        fully_prover_para_cnt += 1;
                    }else{
                        fully_prover_no_para_time += fully_prover_update_time;
                        fully_prover_no_para_cnt += 1;
                    }
                });
            });
            fully_prover_para_time /= fully_prover_para_cnt.try_into().unwrap();
            // fully_prover_no_para_time /= fully_prover_no_para_cnt.try_into().unwrap();
            println!("Fully dynamic Prover Avg Parallel Time: {:?}\n", fully_prover_para_time);
            writeln!(file, "Fully dynamic Prover Avg Parallel Time: {:?}\n", fully_prover_para_time);
            println!("Fully dynamic Prover Avg Non-Parallel Time: {:?}\n", fully_prover_no_para_time);
            writeln!(file, "Fully dynamic Prover Avg Non-Parallel Time: {:?}\n", fully_prover_no_para_time);
        });
        println!("================================================================");
        writeln!(file, "----------------------------------------------------------------");
    }
    
}






pub fn get_masked_proof<E:Pairing>(
    uk: &UpdatingKey<E>,
    raw_a_g1:E::G1, raw_b_g1:E::G1, raw_b_g2:E::G2, raw_c_g1:E::G1
)-> Proof<E>{
    let r: <E as Pairing>::ScalarField = E::ScalarField::rand(&mut ark_std::test_rng());
    let s: <E as Pairing>::ScalarField = E::ScalarField::rand(&mut ark_std::test_rng());
    let a_g1 = raw_a_g1 + uk.pk.delta_g1 * r;
    let b_g1 = raw_b_g1 + uk.pk.delta_g1 * s;
    let b_g2 = raw_b_g2 + uk.pk.vk.delta_g2.into_group() * s;
    let c_g1 = raw_c_g1 + a_g1 * s + b_g1 * r - uk.pk.delta_g1*(r * s);
    Proof { a_g1:a_g1.into_affine(), b_g2: b_g2.into_affine(), c_g1: c_g1.into_affine()} 
}