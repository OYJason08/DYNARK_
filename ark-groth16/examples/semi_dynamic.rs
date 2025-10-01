use ark_bls12_381::Bls12_381;
use ark_ec::{pairing::Pairing,AffineRepr};
use ark_ff::{UniformRand, Zero};
use ark_groth16::{prepare_verifying_key, Groth16};
use ark_relations::r1cs::{
        ConstraintMatrices, Matrix,
    };
use ark_std::{
    rand::{Rng, RngCore, SeedableRng},
    test_rng,
};
use ark_serialize::{CanonicalSerialize,  Compress};
use num_traits::One;
use std::{
    fs::File,
    io::Write,
    time::Instant,
};
use rayon::current_num_threads;
use ark_groth16::instance_generator::{generate_update_once, generate_matrices, generate_instance_witness};

// cargo run --example semi_dynamic --features parallel --release
fn main() {
    // measure_semi_dynamic::<Bls12_381>();
    let mut file= match File::create("semi_bench.txt"){
        Ok(f) => f,
        Err(e)=>{
            eprintln!("Failed to create file: {}", e);
            return;
        }
    };
    // measure_semi_dynamic_trivial::<Bls12_381>();
    println!(" ==================================== ");
    [12,16,20].into_iter().for_each(|log_domain|{
        // bench_semi_dynamic_v2::<Bls12_381>(&mut file, log_domain as usize);
        measure_semi_dynamic_pro::<Bls12_381>(&mut file, log_domain);
        println!(" ==================================== ");
    })
}
/// V2 test
fn measure_semi_dynamic_pro<E: Pairing>(file: &mut File, domain_log_n: usize) {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    let num_assignment = 1 << domain_log_n;
    let num_instance = 1;
    let num_witness = num_assignment - num_instance;
    let num_constraint = num_assignment;
    let num_update = 256;
    println!("Global Cores = {:?}",current_num_threads());
    writeln!(file, "Gloabal Cores = {:?}",current_num_threads());

    let (matrices, ka, kb, ca, cb) =
        generate_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
    let (instance, witness) = generate_instance_witness::<E>(num_instance, num_witness, &mut rng, &ka, &kb, &ca, &cb);

    let setup_start = Instant::now();
    let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynark(matrices.clone(), &mut rng).unwrap();
    let uk = Groth16::<E>::generate_updating_keys(&matrices,&domain, &pk).unwrap();
    let setup_time = setup_start.elapsed();
    println!("Dynark Setup time: {:?}", setup_time);
    writeln!(file, "Dynark Setup time: {:?}", setup_time);

    let prove_start = Instant::now();
    let (proof, mut cache) =
        Groth16::<E>::prove_dynark(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
    let prove_time = prove_start.elapsed();
    println!("Dynark Prove time: {:?}", prove_time);
    writeln!(file, "Dynark Prove time: {:?}", prove_time);

    let preprocess_start = Instant::now();
    Groth16::<E>::process_dynark(&uk, &matrices, &instance, &witness, &mut cache).unwrap();
    let preprocess_time = preprocess_start.elapsed();
    println!("Dynark Preprocess time: {:?}", preprocess_time);
    writeln!(file, "Dynark Preprocess time: {:?}", preprocess_time);

    let verify_old_proof_start = Instant::now();
    let pvk = prepare_verifying_key(&vk);
    let result = Groth16::<E>::verify_dynark(&pvk, &proof, &instance).unwrap();
    let verify_old_time = verify_old_proof_start.elapsed();
    println!("Dynark Old proof verify time: {:?}", verify_old_time);
    println!("Dynark Old proof verification result: {}", result);
    writeln!(file, "Dynark Old proof verify time: {:?}", verify_old_time);
    writeln!(file, "Dynark Old proof verification result: {}", result);
    // println!("Proof_old: {:?}", proof.c_g1.into_group());

    let (instance_update, witness_update) = generate_update_once::<E>(num_constraint, num_update, num_instance, &ka, &kb, &ca, &cb, &instance, &witness, &mut rng);
    let update_start = Instant::now();
    let proof_updated = Groth16::<E>::update_dynark(&uk, &matrices, &instance_update, &witness_update, &cache).unwrap();
    let update_time = update_start.elapsed();
    println!("Dynark Update time: {:?}", update_time);
    writeln!(file, "Dynark Update time: {:?}", update_time);

    let verify_new_start = Instant::now();
    let pvk = prepare_verifying_key(&vk);
    let result = Groth16::<E>::verify_dynark(&pvk, &proof_updated, &instance).unwrap();
    let verify_new_time = verify_new_start.elapsed();
    // println!("Proof_new: {:?}", proof_updated.c_g1.into_group());
    // println!("Proof_new: {:?}", proof_updated.b_g2.into_group());
    // println!("Dynark New proof verify time: {:?}", verify_new_time);
    println!("Dynark New proof verification result: >>>> {} <<<<", result);
    writeln!(file, "Dynark New proof verification result: >>>> {} <<<<", result);
    

    println!("Compressed proof size: {:?}",proof.serialized_size(Compress::Yes));
    println!("unCompressed proof size: {:?}",proof.serialized_size(Compress::No));
    // let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    // pool.install(|| {
    println!("No para now, Cores = {:?}",current_num_threads());
    writeln!(file, "No para now, Cores = {:?}",current_num_threads());
    let update_start = Instant::now();
    let proof_updated = Groth16::<E>::update_dynark(&uk, &matrices, &instance_update, &witness_update, &cache).unwrap();
    let update_time = update_start.elapsed();
    println!("Dynark Update time: {:?}", update_time);
    writeln!(file, "Dynark Update time: {:?}", update_time);

    let verify_new_start = Instant::now();
    let pvk = prepare_verifying_key(&vk);
    let result = Groth16::<E>::verify_dynark(&pvk, &proof_updated, &instance).unwrap();
    let verify_new_time = verify_new_start.elapsed();
    println!("Dynark New proof verify time: {:?}", verify_new_time);
    println!("Dynark New proof verification result: >>>> {} <<<<", result);
    writeln!(file, "Dynark New proof verify time: {:?}", verify_new_time);
    writeln!(file, "Dynark New proof verification result: >>>> {} <<<<", result);
    // });


}















fn measure_semi_dynamic_trivial<E: Pairing>() {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
    let num_assignment = 1 << 4;
    let num_instance = 1;
    let num_witness = num_assignment - num_instance;
    let num_constraint = num_assignment;
    let num_update = 3;

    let matrices =
        generate_trivial_matrices::<E>(num_constraint, num_instance, num_witness, &mut rng);
    let (instance, witness) = generate_instance_witness_trivial::<E>(num_instance, num_witness, &mut rng);

    let setup_start = Instant::now();
    let (pk, vk, domain) = Groth16::<E>::groth16_setup_dynark(matrices.clone(), &mut rng).unwrap();
    let uk = Groth16::<E>::generate_updating_keys(&matrices,&domain, &pk).unwrap();
    let setup_time = setup_start.elapsed();
    println!("Dynark Setup time: {:?}", setup_time);

    let prove_start = Instant::now();
    let (proof, mut cache) =
        Groth16::<E>::prove_dynark(&pk, &matrices, &instance, &witness, &mut rng).unwrap();
    let prove_time = prove_start.elapsed();
    println!("Dynark Prove time: {:?}", prove_time);

    let preprocess_start = Instant::now();
    Groth16::<E>::process_dynark(&uk, &matrices, &instance, &witness, &mut cache).unwrap();
    let pvk = prepare_verifying_key(&vk);
    let preprocess_time = preprocess_start.elapsed();
    println!("Dynark Preprocess time: {:?}", preprocess_time);

    let verify_old_proof_start = Instant::now();
    let result = Groth16::<E>::verify_dynark(&pvk, &proof, &instance).unwrap();
    let verify_old_time = verify_old_proof_start.elapsed();
    println!("Dynark Old proof verify time: {:?}", verify_old_time);
    println!("Dynark Old proof verification result: {}", result);
    // println!("Proof_old: {:?}", proof.c_g1.into_group());

    let witness_update = generate_update_trivial::<E>(num_update, &mut rng);
    let update_start = Instant::now();
    let proof_updated = Groth16::<E>::update_dynark(&uk, &matrices, &[], &witness_update, &cache).unwrap();
    let update_time = update_start.elapsed();
    println!("Dynark Update time: {:?}", update_time);

    let verify_new_start = Instant::now();
    let result = Groth16::<E>::verify_dynark(&pvk, &proof_updated, &instance).unwrap();
    let verify_new_time = verify_new_start.elapsed();
    // println!("Proof_new: {:?}", proof_updated.c_g1.into_group());
    // println!("Proof_new: {:?}", proof_updated.b_g2.into_group());
    // println!("Dynark New proof verify time: {:?}", verify_new_time);
    println!("Dynark New proof verification result: >>>> {} <<<<    [Trivial]", result);
    
    let mut new_witness = witness.clone();
    (witness_update).iter().for_each(|(i,vl)|{new_witness[*i - num_instance]+=vl;});
    // println!("new witness:{:?}", new_witness);
    let (proof, cache) =
        Groth16::<E>::prove_dynark(&pk, &matrices, &instance, &new_witness, &mut rng).unwrap();
    let result: bool = Groth16::<E>::verify_dynark(&pvk, &proof, &instance).unwrap();
    let verify_old_time = verify_old_proof_start.elapsed();
    // println!("Dynark GRT proof verify time: {:?}", verify_old_time);
    println!("Dynark GRT proof verification result: {}", result);

    // let mut sub_proof = proof.clone();
    // sub_proof.a_g1 = proof_updated.a_g1;
    // let result: bool = Groth16::<E>::verify_dynark(&pvk, &sub_proof, &instance).unwrap();
    // println!("Std proof using a_g1 from update verification result: {}", result);

    // let mut sub_proof = proof.clone();
    // sub_proof.b_g2 = proof_updated.b_g2;
    // let result: bool = Groth16::<E>::verify_dynark(&pvk, &sub_proof, &instance).unwrap();
    // println!("Std proof using b_g2 from update verification result: {}", result);

    // let mut sub_proof = proof.clone();
    // sub_proof.c_g1 = proof_updated.c_g1;
    // let result: bool = Groth16::<E>::verify_dynark(&pvk, &sub_proof, &instance).unwrap();
    // println!("Std proof using c_g1 from update verification result: {}", result);
    // println!("Diff: {:?}", (proof.a_g1.into_group() - proof_updated.a_g1.into_group()).into_affine().into_group());
}



fn generate_trivial_matrices<E: Pairing>(
    num_constraint: usize,
    num_instance: usize,
    num_witness: usize,
    rng: &mut impl Rng,
) -> ConstraintMatrices<E::ScalarField> {
    assert_eq!(
        num_constraint,
        num_instance + num_witness,
        "num_constraint must equal num_instance + num_witness"
    );

    let mut a: Matrix<E::ScalarField> = vec![Vec::new(); num_constraint];
    a[0].push((E::ScalarField::one(), 0));
    a[1].push((E::ScalarField::one(), 2));
    a[2].push((E::ScalarField::one(), 3));
    
    let mut b: Matrix<E::ScalarField> = vec![Vec::new(); num_constraint];
    b[0].push((E::ScalarField::one(), 1));
    b[1].push((E::ScalarField::one(), 0));
    b[2].push((E::ScalarField::one(), 3));

    let mut c: Matrix<E::ScalarField> = vec![Vec::new(); num_constraint];
    c[0].push((E::ScalarField::one(), 1));
    c[1].push((E::ScalarField::one(), 2));
    c[2].push((E::ScalarField::one(), 3));

    ConstraintMatrices {
        num_instance_variables: num_instance,
        num_witness_variables: num_witness,
        num_constraints: num_constraint,
        a_num_non_zero: num_constraint,
        b_num_non_zero: num_constraint,
        c_num_non_zero: num_constraint,
        a,
        b,
        c,
    }
}


fn generate_instance_witness_trivial<E: Pairing>(
    num_instance: usize,
    num_witness: usize,
    rng: &mut impl Rng,
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
    let mut instance = vec![E::ScalarField::zero(); num_instance];
    let mut witness = vec![E::ScalarField::zero(); num_witness];
    instance[0] = E::ScalarField::one(); // first instance variable is 1
    witness[0] = E::ScalarField::rand(rng);
    witness[1] = E::ScalarField::rand(rng);
    witness[2] = E::ScalarField::zero();

    (instance, witness)
}

fn generate_update_trivial<E: Pairing>(
    num_update: usize,
    rng: &mut impl Rng,
) -> Vec<(usize, E::ScalarField)> {
    let mut update = Vec::new();
    update.push((1, E::ScalarField::rand(rng)));
    update.push((2, E::ScalarField::rand(rng)));
    update.push((3, E::ScalarField::one()));


    update
}
