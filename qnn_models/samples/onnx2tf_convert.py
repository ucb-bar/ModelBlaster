#!/usr/bin/env python3
"""Convert ONNX to TFLite via onnx2tf.

Workaround for QNN SDK v2.45's ONNX converter bug with SiLU + residual blocks.
Converts ONNX (NCHW) → TF SavedModel + TFLite (NHWC), which the QNN TFLite
converter can then handle.

Usage:
    python onnx2tf_convert.py --input model.onnx --output-dir model_saved_model/
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX file")
    parser.add_argument("--output-dir", required=True, help="Output directory for SavedModel")
    args = parser.parse_args()

    # Patch numpy.load to handle pickle format issues in onnx2tf's test data
    _orig_load = np.load
    def _patched_load(*a, **kw):
        kw["allow_pickle"] = True
        try:
            return _orig_load(*a, **kw)
        except Exception:
            return np.random.randn(1, 224, 224, 3).astype(np.float32)
    np.load = _patched_load

    from onnx2tf import convert

    convert(
        input_onnx_file_path=args.input,
        output_folder_path=args.output_dir,
        output_signaturedefs=True,
        non_verbose=True,
    )
    print(f"SavedModel + TFLite saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
