#!/bin/bash
# Validate checkpoint for domain adaptation
# This script tests if the checkpoint produces valid outputs on both domains

CHECKPOINT="/home/fzhcis/mylab/Open3D-ML/logs/RandLANet_Semantic3DUnified_torch/checkpoint/model_epoch0070_2025-12-03_22-50-31.pth"
CONFIG="/home/fzhcis/mylab/Open3D-ML/ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_forest_full.yml"

echo "========================================"
echo "Checkpoint Validation for DA Training"
echo "========================================"
echo ""
echo "Testing checkpoint: $(basename $CHECKPOINT)"
echo "Config: $(basename $CONFIG)"
echo ""
echo "This will:"
echo "  1. Load your pretrained checkpoint"
echo "  2. Test on Semantic3DUnified (source domain)"
echo "  3. Test on ForestSemantic (target domain)"
echo "  4. Check for NaN/Inf in outputs"
echo "  5. Compare feature distributions"
echo ""
echo "Expected runtime: 2-5 minutes"
echo ""

python scripts/run_tests_fei/validate_checkpoint_da.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --num_batches 1000 \
    --test_source \
    --test_target

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ VALIDATION PASSED"
    echo ""
    echo "Your checkpoint is compatible with DA training."
    echo "You can proceed with training using:"
    echo "  python scripts/run_tests_fei/train_generic.py --config $CONFIG"
else
    echo "❌ VALIDATION FAILED"
    echo ""
    echo "Your checkpoint has issues. Possible causes:"
    echo "  1. grid_size mismatch (config vs checkpoint)"
    echo "  2. Corrupted checkpoint weights"
    echo "  3. Numerical instability"
    echo ""
    echo "Check the detailed output above for specifics."
fi
echo "========================================"

exit $EXIT_CODE
