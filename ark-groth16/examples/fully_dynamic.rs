use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_groth16::{dynamic_cache::FullyCheckpoint, prepare_verifying_key, Groth16};
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng,
};
use std::{
    fs::File,
    io::Write,
    time::Instant,
};
use ark_groth16::instance_generator::{generate_update, generate_matrices, generate_instance_witness};

// cargo run --example fully_dynamic --features parallel --release
fn main() {
    // measure_fully_dynamic::<Bls12_381>();
    let mut file= match File::create("fully_bench.txt"){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    [20].into_iter().for_each(|log_domain|{
        bench_fully_dynamic::<Bls12_381>(&mut file, log_domain as usize);
    })
}

fn bench_fully_dynamic<E: Pairing>(file: &mut File, domain_log_n: usize) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    let num_assignment = 1 << domain_log_n;
    let num_instance = 1;
    let num_witness = num_assignment - num_instance;
    let num_constraint = num_assignment;

    let (matrices, ka, kb, ca, cb) =
        generate_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
    let (mut instance, mut witness) = generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);

    println!(
        "  ================================  \n Starting For Domain of {:?}",
        1 << domain_log_n
    );
    writeln!(
        file,
        "  ================================  \n Starting For Domain of {:?}",
        1 << domain_log_n
    );

    let setup_start = Instant::now();
    let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynark(matrices.clone(), &mut rng).unwrap();
    let setup_time = setup_start.elapsed();
    println!("Dynark Setup time: {:?}", setup_time);
    writeln!(file, "Dynark Setup time: {:?}", setup_time);
    
    let prove_start = Instant::now();
    let (proof, cache) =
    Groth16::<E>::prove_dynark(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
    let prove_time = prove_start.elapsed();
    println!("Dynark Prove time: {:?}", prove_time);
    writeln!(file, "Dynark Prove time: {:?}", prove_time);
    
    
    
    let verify_old_proof_start = Instant::now();
    let pvk = prepare_verifying_key(&vk);
    let result = Groth16::<E>::verify_dynark(&pvk, &proof, &instance).unwrap();
    let verify_old_time = verify_old_proof_start.elapsed();
    println!("Dynark Old proof verify time: {:?}", verify_old_time);
    writeln!(file, "Dynark Old proof verify time: {:?}", verify_old_time);
    println!("Dynark Old proof verification result: {}", result);
    writeln!(file, "Dynark Old proof verification result: {}", result);
    
    let preprocess_start = Instant::now();
    // Groth16::<E>::process_dynark(&uk, &matrices, &instance, &witness, &mut cache);
    let uk = Groth16::<E>::generate_updating_keys(&matrices,&domain, &pk).unwrap();
    let mut checkpoint_using: FullyCheckpoint<E> = FullyCheckpoint::<E>::new(num_constraint, &uk, &matrices, &instance, &witness);
    let mut checkpoint_preparing =  checkpoint_using.clone();
    checkpoint_preparing.prepare_for_reconstruct();
    let preprocess_time = preprocess_start.elapsed();
    println!("Dynark Preprocess time: {:?}", preprocess_time);
    writeln!(file, "Dynark Preprocess time: {:?}", preprocess_time);
    let mut dist_from_last_renew:usize = 0;
    // let mut cur_1: usize = 0;
    // let mut cur_2: usize = 0;
    // let mut cur_3: usize = 0;
    // let mut cur_4: usize = 0;
    // let mut cur_5: usize = 0;
    [
        1 << 0,
        1 << 1,
        1 << 2,
        1 << 3,
        1 << 4,
        1 << 6,
        1<<7,
        1 << 8,
        1 <<10,
    ]
    .into_iter()
    .for_each(|num_update: usize| {
        if (num_update<<1)+1 < num_constraint {
            let rpt = std::cmp::max(1<<12,num_update)/num_update;
            let mut sum_update_time = std::time::Duration::ZERO;
            let mut sum_update_reconstruct_time = std::time::Duration::ZERO;
            let mut sum_update_other_parts_time  = std::time::Duration::ZERO;
            for _ in 0..rpt{

                let (update_instance, update_witness) =
                    generate_update::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &mut instance, &mut witness, &mut rng);
                let update_start = Instant::now();
                let d = (update_instance.len() + update_witness.len() + 1)>>1;
                let mut start_new_checkpoint = false;
                let update_reconstruct_start = Instant::now();
                let mut rem_stp = num_update;
                while rem_stp > 0{
                    // match checkpoint_preparing.cache.status.clone(){
                    //     ReconstructPhase::VecPrepare(_) =>{cur_1+=1;},
                    //     ReconstructPhase::Dft(_) =>{cur_2+=1;},
                    //     ReconstructPhase::PointwiseProduct(_) => {cur_3+=1;},
                    //     ReconstructPhase::IDft(_) => {cur_4+=1;},
                    //     ReconstructPhase::VecIntegrate(_) => {cur_5+=1;},
                    // }
                    let reconstruct_start = Instant::now();
                    let res = checkpoint_preparing.mv_forward(&uk, rem_stp);
                    dist_from_last_renew += rem_stp - res.1;
                    rem_stp = res.1;
                    if res.0{
                        // println!("    VecPrepare: {:?}, Dft: {:?}, PointwiseProduct: {:?}, IDft: {:?}, VecIntegrate: {:?}", cur_1, cur_2, cur_3, cur_4, cur_5);
                        checkpoint_preparing.copy_proof_from(&checkpoint_using);
                        std::mem::swap(&mut checkpoint_preparing,&mut checkpoint_using);
                        checkpoint_preparing.prepare_for_reconstruct();
                        println!("    -------------------------------------------------   Rebuild, ({:?}) steps from last time", dist_from_last_renew);
                        // cur_1 = 0;
                        // cur_2 = 0;
                        // cur_3 = 0;
                        // cur_4 = 0;
                        // cur_5 = 0;
                        dist_from_last_renew = 0;
                        if start_new_checkpoint{
                            break;
                        }
                        start_new_checkpoint = true;
                    }
                    let reconstruct_time: std::time::Duration = reconstruct_start.elapsed();
                    // println!("{:?}   {:?}",reconstruct_time, cat);
                }
                
                let update_reconstruct_time = update_reconstruct_start.elapsed();
                sum_update_reconstruct_time += update_reconstruct_time;
                let update_other_parts_start = Instant::now();
                checkpoint_preparing.trivial_append(&uk, &update_instance, &update_witness);
                checkpoint_using.update_from(&uk, num_instance, num_witness,&update_instance, &update_witness);
                // println!("{:?}, {:?}, {:?}", checkpoint_preparing.raw_a_g1, checkpoint_preparing.raw_b_g1, checkpoint_preparing.raw_c_g1);
                // println!("{:?}, {:?}, {:?}", checkpoint_using.raw_a_g1, checkpoint_using.raw_b_g1, checkpoint_using.raw_c_g1);
                let proof_updated = checkpoint_using.get_proof_with_rs(&uk);
                let update_other_parts_time = update_other_parts_start.elapsed();
                sum_update_other_parts_time += update_other_parts_time;
                let update_time: std::time::Duration = update_start.elapsed();
                sum_update_time += update_time;
                let result = Groth16::<E>::verify_dynark(&pvk, &proof_updated, &instance).unwrap();
                if !result {
                    panic!(" Verify Failed");
                }
                
            }
            sum_update_time /= rpt.try_into().unwrap();
            sum_update_reconstruct_time /= rpt.try_into().unwrap();
            sum_update_other_parts_time /= rpt.try_into().unwrap();
            println!(
                "        Total Update time for ({:?})-slot-update:  {:?}",
                num_update, sum_update_time,
            );
            let _ = writeln!(
                file,
                "        Total Update time for ({:?})-slot-update:  {:?}",
                num_update, sum_update_time,
            );
            println!(
                "        Contains reconstruction tread using ({:?})",
                sum_update_reconstruct_time
            );
            let _ = writeln!(
                file,
                "        Contains reconstruction tread using ({:?})",
                sum_update_reconstruct_time
            );
            println!(
                "        Contains other part of updates using ({:?})",
                sum_update_other_parts_time
            );
            let _ = writeln!(
                file,
                "        Contains other part of updates using ({:?})",
                sum_update_other_parts_time
            );
        }
    });
    println!(" ----------------------------------- \n");
    let _ = writeln!(file, " ----------------------------------- \n");
}
