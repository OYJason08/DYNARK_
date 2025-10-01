use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_groth16::{prepare_verifying_key, Groth16};
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng,
};
use ark_serialize::{CanonicalSerialize,Compress};
use std::{
    fs::File,
    io::Write,
    time::Instant,
};
use ark_groth16::instance_generator::{generate_update_once, generate_matrices, generate_instance_witness};
use ark_groth16::dynamic_cache::{FullyCheckpoint};
use ark_groth16::instance_generator::{generate_update};
use std::time::Duration;
use rayon::{current_num_threads,ThreadPoolBuilder};
// cargo run --example bench_v1 --features parallel --release
fn main() {
    println!("Global Cores = {:?}",current_num_threads());
    // Test_Setup::<Bls12_381>(20,10);
    Test_Preprocess::<Bls12_381>(3);
    // Test_ProveTime_d1_moveN::<Bls12_381>(1<<12);
    // Test_ProveTime_FixN_move_d::<Bls12_381>(20,10);
}

/// Test Setup
fn Test_Setup<E: Pairing>(log_n: usize, repeat_time:usize){
    let mut file= match File::create(format!("Setup_Time({}).txt", 1<<log_n)){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    let num_assignment = 1 << log_n;
    let num_instance = 1;
    let num_witness = num_assignment - num_instance;
    let num_constraint = num_assignment;

    let (matrices, ka, kb, ca, cb) =
        generate_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
    

    writeln!(file, "Start For N = {:?}\n", num_assignment);
    let groth16_processing_time = (0..repeat_time).into_iter().map(|_|{
        let prove_start = Instant::now();
        let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynarc(matrices.clone(), &mut rng).unwrap();
        prove_start.elapsed()
    }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
    println!("Standard Groth16 Setup Avg Time: {:?}\n----------------------------------------------------------------", groth16_processing_time);
    writeln!(file, "Standard Groth16 Setup Avg Time: {:?}\n", groth16_processing_time);


    let dynark_preprocessing_time = (0..repeat_time).into_iter().map(|_|{
        let prove_start = Instant::now();
        let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynarc(matrices.clone(), &mut rng).unwrap();
        let _ = Groth16::<E>::generate_updating_keys(&matrices, &domain, &pk);
        prove_start.elapsed()
    }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
    println!("Our Dynark Setup Avg Time: {:?}\n================================================================", dynark_preprocessing_time);
    writeln!(file, "Our Dynark Setup Avg Time: {:?}\n----------------------------------------------------------------", dynark_preprocessing_time);
}

/// Test Preprocessing
fn Test_Preprocess<E: Pairing>(repeat_time:usize) {
    let mut file= match File::create("Preprocessing_Time.txt"){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    (11..23).into_iter().for_each(|domain_log_n: usize|{
        let num_assignment = 1 << domain_log_n;
        let num_instance = 1;
        let num_witness = num_assignment - num_instance;
        let num_constraint = num_assignment;

        let (matrices, ka, kb, ca, cb) =
            generate_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
        let (instance, witness) = generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);
        let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynarc(matrices.clone(), &mut rng).unwrap();
        let uk = Groth16::<E>::generate_updating_keys(&matrices, &domain, &pk).unwrap();
        let (_, mut cache) =
            Groth16::<E>::prove_dynarc(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
        
        println!("Start For N = {:?}\n", num_assignment);
        writeln!(file, "Start For N = {:?}\n", num_assignment);

        let dynark_preprocessing_time = (0..repeat_time).into_iter().map(|_|{
            let pre_processing_start = Instant::now();
            Groth16::<E>::process_dynarc(&uk, &matrices, &instance, &witness, &mut cache).unwrap();
            let preprocessing_time = pre_processing_start.elapsed();
            preprocessing_time
        }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
        println!("Parallel Dynark Preprocessing Avg Time: {:?}\n", dynark_preprocessing_time);
        writeln!(file, "Parallel Dynark Preprocessing Avg Time: {:?}\n", dynark_preprocessing_time);

        if domain_log_n<=18{
            let dynark_preprocessing_time = (0..repeat_time).into_iter().map(|_|{
                let mut preprocessing_time = Duration::ZERO;
                let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
                pool.install(||{
                    let pre_processing_start = Instant::now();
                    Groth16::<E>::process_dynarc(&uk, &matrices, &instance, &witness, &mut cache).unwrap();
                    preprocessing_time = pre_processing_start.elapsed();
                });
                preprocessing_time
            }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
            println!("No-Parallel Dynark Preprocessing Time: {:?}\n", dynark_preprocessing_time);
            writeln!(file, "No-Parallel Dynark Preprocessing Time: {:?}\n", dynark_preprocessing_time);
        }

        println!("----------------------------------------------------------------");
        writeln!(file, "================================================================");
    });

}


/// Test Prover Time
fn Test_ProveTime_d1_moveN<E: Pairing>(repeat_time:usize) {
    let mut file= match File::create("Prover_Time_Fix_d=1_Move_N_test.txt"){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    

    (10..23).into_iter().for_each(|log_n: usize|{
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

        println!("                              Got Cached Quotient");
        Groth16::<E>::process_dynarc(&uk, &matrices, &instance, &witness, &mut cache).unwrap();

        let instance_copy = instance.clone();
        let witness_copy = witness.clone();

        let dist_from_last_renew:usize = 0;
        let num_update:usize = 1;
        println!("Start For N = {:?} with d = {:?}\n", num_assignment, num_update);
        writeln!(file, "Start For N = {:?} with d = {:?}\n", num_assignment, num_update);


// Fully Part
        if (log_n&1)==0{

            let mut checkpoint_using: FullyCheckpoint<E> = FullyCheckpoint::<E>::new(num_constraint, &uk, &matrices, &instance, &witness);
            let mut checkpoint_preparing =  checkpoint_using.clone();
            println!("                              Got Fully Checkpoint");
            let fully_dynamic_prover_time = (0..repeat_time).into_iter().map(|_|{
                let (update_instance, update_witness) =
                    generate_update::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng);
                let update_reconstruct_start = Instant::now();
                let mut start_new_checkpoint = false;
                let mut rem_stp = (num_update+1)>>1;
                while rem_stp > 0{
                    // dist_from_last_renew += 1;
                    // let reconstruct_start = Instant::now();
                    let res = checkpoint_preparing.mv_forward(&uk, rem_stp);
                    rem_stp = res.1;
                    if res.0{
                        checkpoint_preparing.copy_proof_from(&checkpoint_using);
                        std::mem::swap(&mut checkpoint_preparing,&mut checkpoint_using);
                        checkpoint_preparing.prepare_for_reconstruct();
                        // println!("    -------------------------------------------------   Rebuild, ({:?}) steps from last time", dist_from_last_renew);
                        // dist_from_last_renew = 0;
                        if start_new_checkpoint{
                            break;
                        }
                        start_new_checkpoint = true;
                    }
                    // let reconstruct_time: std::time::Duration = reconstruct_start.elapsed();
                    // println!("{:?}   {:?}",reconstruct_time, cat);
                }
                
                let update_reconstruct_time = update_reconstruct_start.elapsed();
                let update_other_parts_start = Instant::now();
                checkpoint_preparing.trivial_append(&uk, &update_instance, &update_witness);
                checkpoint_using.update_from(&uk, num_instance, num_witness,&update_instance, &update_witness);
                // println!("{:?}, {:?}, {:?}", checkpoint_preparing.raw_a_g1, checkpoint_preparing.raw_b_g1, checkpoint_preparing.raw_c_g1);
                // println!("{:?}, {:?}, {:?}", checkpoint_using.raw_a_g1, checkpoint_using.raw_b_g1, checkpoint_using.raw_c_g1);
                let proof_updated = checkpoint_using.get_proof_with_rs(&uk);
                let update_other_parts_time = update_other_parts_start.elapsed();
                let result = Groth16::<E>::verify_dynarc(&pvk, &proof_updated, &instance).unwrap();
                let prove_time = update_reconstruct_time + update_other_parts_time;
                // println!("        Fully Dynamic Prover time: {:?},    Result = {:?}", prove_time, result);
                if !result{
                    panic!("Fully Prover Fail");
                }
                prove_time
            }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
            println!("Fully dynamic Prover Avg Time: {:?}\n", fully_dynamic_prover_time);
            writeln!(file, "Fully dynamic Prove Avg Time: {:?}\n", fully_dynamic_prover_time);
        }

// Semi Part
        let semi_prover_time = (0..repeat_time).into_iter().map(|_|{
            let (instance_update, witness_update) =
                generate_update_once::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &instance_copy, &witness_copy, &mut rng);
            let semi_prover_start = Instant::now();
            let proof_updated = Groth16::<E>::update_dynarc(&uk, &matrices, &instance_update, &witness_update, &cache).unwrap();
            let semi_prover_update_time = semi_prover_start.elapsed();
            let result = Groth16::<E>::verify_dynarc(&pvk, &proof_updated, &instance_copy).unwrap();
            // println!("Dynarc New proof verification result: >>>> {} <<<<", result);
            if !result{
                panic!("Semi Prover Fail");
            }
            semi_prover_update_time
        }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
        println!("Semi dynamic Prover Avg Time: {:?}\n================================================================", semi_prover_time);
        writeln!(file, "Semi dynamic Prover Avg Time: {:?}\n----------------------------------------------------------------", semi_prover_time);
    });

}


/// Test Prover Time
fn Test_ProveTime_FixN_move_d<E: Pairing>(log_n: usize, repeat_time:usize) {
    let mut file= match File::create(format!("Para_Prover_Time_N=(2^{})_move_d_test_Small.txt",log_n)){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
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
    let uk = Groth16::<E>::generate_updating_keys(&matrices,&domain, &pk).unwrap();
    println!("                              Got Update Key");
    let (_, mut cache) =
        Groth16::<E>::prove_dynarc(&pk, &matrices, &instance, &witness, &mut rng).unwrap();

    println!("                              Got Cached Quotient");
    Groth16::<E>::process_dynarc(&uk, &matrices, &instance, &witness, &mut cache).unwrap();

    let instance_copy = instance.clone();
    let witness_copy = witness.clone();

    let mut checkpoint_using: FullyCheckpoint<E> = FullyCheckpoint::<E>::new(num_constraint, &uk, &matrices, &instance, &witness);
    let mut checkpoint_preparing =  checkpoint_using.clone();
    println!("                              Got Fully Checkpoint");

    let dist_from_last_renew:usize = 0;

    (0..std::cmp::min(9, log_n-1)).into_iter().for_each(|log_d: usize|{
        let num_update:usize = 1<<log_d;
        println!("Start For d = {:?} with N = {:?}\n", num_update, num_assignment);
        writeln!(file, "Start For d = {:?} with N = {:?}\n", num_update, num_assignment);
        
        
        // Fully Part
        if log_d < 11{
            let real_repeat_time = std::cmp::max(repeat_time, (1<<((log_n+6)>>1))>>log_d);
            let fully_dynamic_prover_time = (0..real_repeat_time).into_iter().map(|_|{
                let (update_instance, update_witness) =
                    generate_update::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng);
                let update_reconstruct_start = Instant::now();
                let mut start_new_checkpoint = false;
                let mut rem_stp = (num_update+1)>>1;
                while rem_stp > 0{
                    // dist_from_last_renew += 1;
                    // let reconstruct_start = Instant::now();
                    let res = checkpoint_preparing.mv_forward(&uk, rem_stp);
                    rem_stp = res.1;
                    if res.0{
                        checkpoint_preparing.copy_proof_from(&checkpoint_using);
                        std::mem::swap(&mut checkpoint_preparing,&mut checkpoint_using);
                        checkpoint_preparing.prepare_for_reconstruct();
                        // println!("    -------------------------------------------------   Rebuild, ({:?}) steps from last time", dist_from_last_renew);
                        // dist_from_last_renew = 0;
                        if start_new_checkpoint{
                            break;
                        }
                        start_new_checkpoint = true;
                    }
                    // let reconstruct_time: std::time::Duration = reconstruct_start.elapsed();
                    // println!("{:?}   {:?}",reconstruct_time, cat);
                }
                
                let update_reconstruct_time = update_reconstruct_start.elapsed();
                let update_other_parts_start = Instant::now();
                checkpoint_preparing.trivial_append(&uk, &update_instance, &update_witness);
                checkpoint_using.update_from(&uk, num_instance, num_witness,&update_instance, &update_witness);
                // println!("{:?}, {:?}, {:?}", checkpoint_preparing.raw_a_g1, checkpoint_preparing.raw_b_g1, checkpoint_preparing.raw_c_g1);
                // println!("{:?}, {:?}, {:?}", checkpoint_using.raw_a_g1, checkpoint_using.raw_b_g1, checkpoint_using.raw_c_g1);
                let proof_updated = checkpoint_using.get_proof_with_rs(&uk);
                let update_other_parts_time = update_other_parts_start.elapsed();
                let result = Groth16::<E>::verify_dynarc(&pvk, &proof_updated, &instance).unwrap();
                let prove_time = update_reconstruct_time + update_other_parts_time;
                // println!("        Fully Dynamic Prover time: {:?},    Result = {:?}", prove_time, result);
                if !result{
                    panic!("Fully Prover Fail");
                }
                prove_time
            }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
            println!("Fully dynamic Prover Avg Time: {:?}\n", fully_dynamic_prover_time);
            writeln!(file, "Fully dynamic Prove Avg Time: {:?}\n", fully_dynamic_prover_time);
        }
// Semi Part
        let semi_prover_time = (0..repeat_time).into_iter().map(|_|{
            let (instance_update, witness_update) =
                generate_update_once::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &instance_copy, &witness_copy, &mut rng);
            let semi_prover_start = Instant::now();
            let proof_updated = Groth16::<E>::update_dynarc(&uk, &matrices, &instance_update, &witness_update, &cache).unwrap();
            let semi_prover_update_time = semi_prover_start.elapsed();
            let result = Groth16::<E>::verify_dynarc(&pvk, &proof_updated, &instance_copy).unwrap();
            // println!("Dynarc New proof verification result: >>>> {} <<<<", result);
            if !result{
                panic!("Semi Prover Fail");
            }
            semi_prover_update_time
        }).sum::<Duration>() / repeat_time.try_into().unwrap();//.collect::<Vec<Duration> >();
        println!("Semi dynamic Prover Avg Time: {:?}\n================================================================", semi_prover_time);
        writeln!(file, "Semi dynamic Prover Avg Time: {:?}\n----------------------------------------------------------------", semi_prover_time);
    });

}