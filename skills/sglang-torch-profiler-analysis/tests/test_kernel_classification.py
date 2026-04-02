import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_sglang_llm_torch_profile as llm  # noqa: E402
import analyze_sglang_profiler_overlap as overlap  # noqa: E402


CUTLASS_FP8_GEMM = (
    "_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10collective"
    "13CollectiveMmaINS1_35MainloopSm100TmaUmmaWarpSpecializedILi13ELi2ELi4ENS5_IJNS4_1CILi1EEE"
    "NSA_ILi4EEESB_EEEEENS5_IJNSA_ILi64EEESF_NSA_ILi128EEEEEENS_12float_e4m3_tENS5_IJlSB_lEEE"
    "SI_SJ_NS4_8TiledMMAINS4_8MMA_AtomIJNS4_10MMA_TraitsINS4_19SM100_MMA_F8F6F4_SSEJSI_SI_fSF_SF_"
    "NS4_17integral_constantINS4_4UMMA5MajorELSQ_0EEESR_NSO_INSP_7ScaleInELSS_0EEEST_EEEEEENS4_6Lay"
    "outINS5_IJSB_SB_SB_EEENS5_IJNSA_ILi0EEESY_SY_EEEEENS5_IJNS4_10UnderscoreES11_S11_EEEEENS4_23S"
    "M90_TMA_LOAD_MULTICASTENS4_14ComposedLayoutINS4_7SwizzleILi3ELi4ELi3EEENS4_18smem_ptr_flag_bi"
    "tsILi8EEENSW_INS5_IJNSA_ILi8EEESG_EEENS5_IJSG_SB_EEEEEEEvNS4_8identityENS4_13SM90_TMA_LOADES1"
    "E_vS1F_EENS_8epilogue10collective18CollectiveEpilogueINS1I_23Sm100TmaWarpSpecializedILi1ELi1E"
    "Li32ELb0ELb1EEEJSH_NS5_IJNSW_ISF_SB_EES1N_EEEvSJ_NS_6half_tESJ_NS1I_6fusion15Sm90TreeVisitorIN"
    "S1Q_11Sm90ComputeINS_10multipliesES1P_fLNS_15FloatRoundStyleE2EvEEJNS1Q_16Sm90ColBroadcastILi0"
    "ESH_ffNS5_IJSB_SY_SY_EEELi4ELb1EEENS1R_INS1S_IS1T_ffLS1U_2EvEEJNS1Q_16Sm90RowBroadcastILi0ESH_f"
    "fNS5_IJSY_SB_SY_EEELi4ELb1EEENS1Q_12Sm90AccFetchEEEEEEENS4_5SM1004TMEM4LOAD26SM100_TMEM_LOAD_1"
    "6dp256b8xES1G_NS15_IS17_NS18_ILi16EEENSW_INS5_IJS1A_SF_EEENS5_IJSF_SB_EEEEEEENS4_17SM75_U32x4_"
    "LDSM_NENS4_14SM90_TMA_STOREES2E_NS4_17SM90_U32x4_STSM_NENS4_39AutoVectorizingCopyWithAssumedAl"
    "ignmentILi128EEEEEEvvEEEEvNT_6ParamsE"
)

FLOOR_ELEMENTWISE = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, "
    "at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::"
    "operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*"
    ", 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor"
    "_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operat"
    "or()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>)"
)

COPY_KERNEL = (
    "void at::native::vectorized_elementwise_kernel<8, at::native::float16_copy_kernel>"
)

FLASHINFER_ACT = "void flashinfer::activation::act_and_mul_kernel<__half, &"

FLASHINFER_NORM = "void flashinfer::norm::FusedAddRMSNormKernel<8u, __half>"


class TestKernelClassification(unittest.TestCase):
    def test_llm_breakdown_classifies_cutlass_fp8_linear_as_gemm(self):
        self.assertEqual(llm.classify_kernel(CUTLASS_FP8_GEMM), "gemm")

    def test_llm_breakdown_classifies_floor_kernel_as_elementwise(self):
        self.assertEqual(llm.classify_kernel(FLOOR_ELEMENTWISE), "elementwise")

    def test_llm_breakdown_classifies_copy_kernel_as_memory(self):
        self.assertEqual(llm.classify_kernel(COPY_KERNEL), "memory")

    def test_llm_breakdown_preserves_flashinfer_activation_and_norm(self):
        self.assertEqual(llm.classify_kernel(FLASHINFER_ACT), "activation")
        self.assertEqual(llm.classify_kernel(FLASHINFER_NORM), "norm")

    def test_overlap_classifies_cutlass_fp8_linear_as_compute(self):
        self.assertEqual(overlap.classify_kernel(CUTLASS_FP8_GEMM), "compute")

    def test_overlap_classifies_floor_kernel_as_elementwise(self):
        self.assertEqual(overlap.classify_kernel(FLOOR_ELEMENTWISE), "elementwise")


if __name__ == "__main__":
    unittest.main()
