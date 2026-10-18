use std::sync::Arc;

use symtn_core::{BlockKey, FusionTreeHomSpace, SU2FusionRule};
use symtn_tensors::{
    prepare_tensorcontract_fusion_plan_dyn, BoundDynamicFusionMapSpace, ContractDestinationInit,
    DynamicFusionMapSpace, FusionOperand, OutputAxisOrder, RuleIdentity,
    TensorContractFusionExecutionContext, TensorContractSpec, TreeTransformExecutionContext,
};

fn main() {
    let rule = SU2FusionRule;
    let provider = Arc::new(rule);
    let bind = |space: DynamicFusionMapSpace| {
        BoundDynamicFusionMapSpace::bind_multiplicity_free(space, Arc::clone(&provider)).unwrap()
    };
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2), (1, 2), (1, 2)], [(1, 2)]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 2), (1, 2), (1, 2)]);
    let axes = TensorContractSpec::with_default_output_order(&[0, 1, 2], &[1, 2, 3]);
    let tensorkit_axes =
        TensorContractSpec::new(&[0, 1, 2], &[1, 2, 3], OutputAxisOrder::from_axes(&[1, 0]));
    let lhs_core_hom = lhs_hom
        .permute(&rule, &[3], &[0, 1, 2])
        .expect("valid lhs core tree-pair transform");
    let rhs_core_hom = rhs_hom
        .permute(&rule, &[1, 2, 3], &[0])
        .expect("valid rhs core tree-pair transform");
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &lhs_hom,
        &rhs_hom,
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &[0, 1],
        1,
    )
    .unwrap();

    println!("lhs_keys");
    for (i, key) in lhs_hom.fusion_tree_keys(&rule).iter().enumerate() {
        println!("{i}: {key:?}");
    }
    println!("rhs_keys");
    for (i, key) in rhs_hom.fusion_tree_keys(&rule).iter().enumerate() {
        println!("{i}: {key:?}");
    }
    println!("lhs_core_keys");
    for (i, key) in lhs_core_hom.fusion_tree_keys(&rule).iter().enumerate() {
        println!("{i}: {key:?}");
    }
    println!("rhs_core_keys");
    for (i, key) in rhs_core_hom.fusion_tree_keys(&rule).iter().enumerate() {
        println!("{i}: {key:?}");
    }

    let lhs_space = DynamicFusionMapSpace::from_final_homspace(&rule, lhs_hom).unwrap();
    let rhs_space = DynamicFusionMapSpace::from_final_homspace(&rule, rhs_hom).unwrap();
    let lhs_core_space = DynamicFusionMapSpace::from_final_homspace(&rule, lhs_core_hom).unwrap();
    let rhs_core_space = DynamicFusionMapSpace::from_final_homspace(&rule, rhs_core_hom).unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    print_structure("lhs_structure", lhs_space.structure());
    print_structure("rhs_structure", rhs_space.structure());
    print_structure("lhs_core_structure", lhs_core_space.structure());
    print_structure("rhs_core_structure", rhs_core_space.structure());
    print_structure("dst_structure", dst_space.structure());

    let lhs_data = (0..32)
        .map(|index| 1.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| -3.0 + 0.25 * index as f64)
        .collect::<Vec<_>>();
    let mut lhs_core_data = vec![0.0; lhs_core_space.required_len().unwrap()];
    let mut rhs_core_data = vec![0.0; rhs_core_space.required_len().unwrap()];
    let mut dst_data = vec![2.0, -1.0, 4.0, -3.0];
    let lhs_bound = bind(lhs_space.clone());
    let rhs_bound = bind(rhs_space.clone());
    let dst_bound = bind(dst_space);
    let plan =
        prepare_tensorcontract_fusion_plan_dyn(&dst_bound, &lhs_bound, &rhs_bound, axes).unwrap();
    let mut tree_context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    tree_context
        .tree_transform_dyn_into(
            &rule,
            plan.lhs_transform().clone(),
            lhs_core_space.structure(),
            lhs_space.structure(),
            &mut lhs_core_data,
            &lhs_data,
            1.0,
            0.0,
        )
        .unwrap();
    tree_context
        .tree_transform_dyn_into(
            &rule,
            plan.rhs_transform().clone(),
            rhs_core_space.structure(),
            rhs_space.structure(),
            &mut rhs_core_data,
            &rhs_data,
            1.0,
            0.0,
        )
        .unwrap();
    println!("lhs_core_data {lhs_core_data:?}");
    println!("rhs_core_data {rhs_core_data:?}");
    TensorContractFusionExecutionContext::<_, RuleIdentity>::default()
        .tensorcontract_planned_into(
            &dst_bound,
            &mut dst_data,
            (FusionOperand::direct(lhs_bound.space()), &lhs_data),
            (FusionOperand::direct(rhs_bound.space()), &rhs_data),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &[0, 1],
            -1.5,
            ContractDestinationInit::Axpby(0.25),
        )
        .unwrap();
    println!("dst_data {dst_data:?}");
    println!("dst_checksum {:.12}", dst_data.iter().copied().sum::<f64>());

    let tensorkit_dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_space.homspace(),
        rhs_space.homspace(),
        tensorkit_axes.lhs_contracting_axes(),
        tensorkit_axes.rhs_contracting_axes(),
        &[1, 0],
        1,
    )
    .unwrap();
    let tensorkit_dst_space =
        DynamicFusionMapSpace::from_final_homspace(&rule, tensorkit_dst_hom).unwrap();
    print_structure(
        "tensorkit_order_dst_structure",
        tensorkit_dst_space.structure(),
    );
    let mut tensorkit_order_dst = vec![2.0, -1.0, 4.0, -3.0];
    TensorContractFusionExecutionContext::<_, RuleIdentity>::default()
        .tensorcontract_planned_into(
            &bind(tensorkit_dst_space.clone()),
            &mut tensorkit_order_dst,
            (FusionOperand::direct(lhs_bound.space()), &lhs_data),
            (FusionOperand::direct(rhs_bound.space()), &rhs_data),
            tensorkit_axes.lhs_contracting_axes(),
            tensorkit_axes.rhs_contracting_axes(),
            &[1, 0],
            -1.5,
            ContractDestinationInit::Axpby(0.25),
        )
        .unwrap();
    println!("tensorkit_order_dst_data {tensorkit_order_dst:?}");
    println!(
        "tensorkit_order_dst_checksum {:.12}",
        tensorkit_order_dst.iter().copied().sum::<f64>()
    );
}

fn print_structure(label: &str, structure: &symtn_core::BlockStructure) {
    println!("{label}");
    for i in 0..structure.block_count() {
        let block = structure.block(i).unwrap();
        match block.key() {
            BlockKey::FusionTree(key) => {
                println!(
                    "{i}: offset={} shape={:?} key={key:?}",
                    block.offset(),
                    block.shape()
                );
            }
            key => {
                println!(
                    "{i}: offset={} shape={:?} key={key:?}",
                    block.offset(),
                    block.shape()
                );
            }
        }
    }
}
