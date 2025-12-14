# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve, root_scalar
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

APP_TITLE = "Constant Heat-Flux TEG Optimization Tool"

# ============================ Utility: Paths & Custom Library Files ============================

def app_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

CUSTOM_LIB_NAME = "Custom_Material_Library.txt"
CUSTOM_LIB_PATH = os.path.join(app_base_dir(), CUSTOM_LIB_NAME)

CUSTOM_HEADER = (
"# Custom material library parameter file (UTF-8)\n"
"# This is a JSON array (after removing lines beginning with #, it becomes valid JSON).\n"
"# Fields for each record:\n"
"#   name: material name; type: 'P' or 'N'; Tmax: maximum safe temperature (K)\n"
"#   S / k / rho: each is a raw text block, with the same format as the main UI input boxes, e.g.:\n"
"#       \"Ts=[...];\\ns_t=[...]\", \"Tk=[...];\\nk_t=[...]\", \"Tr=[...];\\nr_t=[...]\".\n"
"# You can also write via the program UI using \"Save to Custom Library\"; or edit this file manually; it will be loaded automatically on next startup.\n"
"# ------------------------------------------------------------------------------\n"
)

CUSTOM_EXAMPLE = [
    {
        "name": "Custom Example Material (P, 600K)",
        "type": "P",
        "Tmax": 600,
        "S": "Ts=[300 350 400 450 500 550 600];\ns_t=[0.00015 0.00017 0.00019 0.0002 0.000205 0.00021 0.000215];",
        "k": "Tk=[300 350 400 450 500 550 600];\nk_t=[1.5 1.45 1.4 1.35 1.3 1.28 1.25];",
        "rho": "Tr=[300 350 400 450 500 550 600];\nr_t=[1.2e-5 1.3e-5 1.4e-5 1.5e-5 1.6e-5 1.7e-5 1.8e-5];"
    }
]



def ensure_custom_lib():
    if not os.path.exists(CUSTOM_LIB_PATH):
        try:
            with open(CUSTOM_LIB_PATH, "w", encoding="utf-8") as f:
                f.write(CUSTOM_HEADER)
                json.dump(CUSTOM_EXAMPLE, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to create custom library file: {e}")

def load_custom_materials():
    ensure_custom_lib()
    try:
        with open(CUSTOM_LIB_PATH, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if not ln.strip().startswith("#")]
            txt = "".join(lines).strip()
            if not txt:
                return []
            return json.loads(txt)
    except Exception:
        return []

def save_material_to_custom_lib(entry):
    mats = load_custom_materials()
    names = [m.get("name", "") for m in mats]
    if entry["name"] in names:
        mats[names.index(entry["name"])] = entry
    else:
        mats.append(entry)
    with open(CUSTOM_LIB_PATH, "w", encoding="utf-8") as f:
        f.write(CUSTOM_HEADER)
        json.dump(mats, f, ensure_ascii=False, indent=2)

# ============================ Default Pasted Data ============================

default_rawdata = {
    'n': {
        'S': 'Ts=[300\t325\t350\t375\t400\t425\t450\t475\t500\t525\t550\t575\t600];\ns_t=-1*[0.00018\t0.00019\t0.0002\t0.000215\t0.000225\t0.000235\t0.000245\t0.00025\t0.00026\t0.00027\t0.000275\t0.00028\t0.00029];',
        'k': 'Tk=[301.713 327.231 348.411 372.674 398.363 422.341 449.686 476.175 503.406 525.043 548.506 575.395 603.482];\nk_t=[1.29464 1.2312 1.17176 1.13584 1.08943 1.04265 0.998764 0.962122 0.933084 0.906232 0.893138 0.901403 0.908578];',
        'rho': 'Tr=[298.559 324.129 345.305 367.793 390.508 409.913 432 453.801 476.573 499.003 518.691 538.666 560.18 578.556 600.012];\nr_t=1e-5*[1.74657 1.76272 1.76739 1.8092 1.88678 1.98259 2.08313 2.20594 2.32403 2.41376 2.59531 2.73973 2.9206 3.08325 3.34918];'
    },
    'p': {
        'S': 'Ts=[303.2138 323.1763 347.9943 372.9239 398.155 423.08 448.041 473.189 498.184 523.289 548.317];\ns_t=[0.000174107 0.000184617 0.000194841 0.00020202 0.000206271 0.000207911 0.000206304 0.000201856 0.000194363 0.000183564 0.000169952];',
        'k': 'Tk=[266.4579 286.784 311.6612 336.9557 361.605 386.255 411.514 436.887 461.466 486.615 511.841];\nk_t=[1.31396 1.2622 1.20815 1.15785 1.10293 1.059 1.03968 1.05336 1.09714 1.17421 1.2643];',
        'rho': 'Tr=[266.8227 286.7758 312.0394 337.0371 362.148 387.373 412.218 437.215 462.325 487.245 512.659];\nr_t=[1.08659E-05 1.20764E-05 1.33164E-05 1.42959E-05 1.49135E-05 1.53139E-05 1.54394E-05 1.52247E-05 0.000014706 1.39123E-05 1.29595E-05];'
    },
    'single': {
        'S': 'Ts=[299.749 321.905 372.187 423.054 473.258 522.509 572.647 622.779 673.119 722.762 772.814 822.511 872.934];\ns_t=[3.06E-04 3.13E-04 3.34E-04 3.54E-04 3.72E-04 3.93E-04 4.19E-04 4.38E-04 4.46E-04 4.22E-04 3.69E-04 3.29E-04 3.45E-04];',
        'k': 'Tk=[299.915 323.739 373.399 423.573 473.44 523.616 574.294 623.264 673.032 723.201 772.975 823.057 873.332];\nk_t=[1.02248 0.905408 0.749523 0.699682 0.596819 0.565005 0.539555 0.477625 0.383245 0.288441 0.244751 0.258112 0.213574];',
        'rho': 'Tr=[299.367 322.702 373.648 422.239 472.217 522.106 572.644 623.03 673.203 723.179 773.179 822.821 873.449];\nr_t=1./[900.724 1178.77 1988.39 2752.21 3224.05 3205.98 2769.96 2190 1876.67 2247.09 3393.68 4801.99 4883.7];'
    }
}

# ============================ Built-in Material Library ============================
# (Please paste the full BUILTIN_MATERIALS list here if needed, keeping it short for this fix)
BUILTIN_MATERIALS = [
        # ---------------- SiGe (1123K) ----------------
    {
        "name": "Si80Ge20P2 (N-Type 1123K)",
        "type": "N",
        "Tmax": 1123,
        "S": "Ts=[324.6134 335.5687 346.1845 351.7045 364.0182 381.173 398.837 423.635 458.114 484.951 505.334 523.255 551.707 582.792 625.683 685.731 701.274 719.536 739.92 756.993 770.413 784.173 792.667 807.192 818.914 829.106 843.801 857.221 867.075 875.824 888.31 904.28 918.89 926.79 938.852 955.757 984.641 999.083 1013.016 1030.602 1044.959 1055.919 1060.082 1068.407 1075.459 1091.092 1103.581 1115.561 1127.37 1140.03 1156.767 1161.95 1171.806];\n"
             "s_t=[-0.000107723 -0.00011135 -0.000114612 -0.00011658 -0.000121022 -0.000126871 -0.000132804 -0.00014107 -0.000153048 -0.000160696 -0.000166291 -0.00017093 -0.0001781 -0.000186057 -0.000196432 -0.000210433 -0.000213751 -0.000217266 -0.000221314 -0.000224604 -0.000227275 -0.000229608 -0.000231014 -0.000233404 -0.00023509 -0.00023689 -0.000239083 -0.000241332 -0.000242822 -0.000244115 -0.000245577 -0.000247573 -0.000249372 -0.000250244 -0.000251087 -0.000252071 -0.000253223 -0.000253279 -0.000252997 -0.000252772 -0.000252631 -0.000252096 -0.000251955 -0.000251955 -0.000251646 -0.000250408 -0.000249423 -0.000248495 -0.000247454 -0.000246357 -0.00024512 -0.000244557 -0.000243657];",
        "rho": "Tr=[314.6892 340.3433 368.0362 383.666 395.219 404.054 409.575 416.541 423.252 426.31 430.303 435.739 443.554 450.095 460.459 469.718 487.132 499.11 514.231 528.587 559.083 584.312 601.981 616.167 632.392 658.386 679.113 727.193 743.333 760.833 780.71 798.464 812.566 843.571 865.148 904.224 925.546 944.404 959.015 987.047 1009.304 1036.062 1054.581 1067.068 1082.868 1094.081 1124.747 1139.783 1155.753 1166.201 1172.742];\n"
               "r_t=[8.53679E-06 8.8836E-06 9.29472E-06 9.54636E-06 9.73805E-06 9.89227E-06 9.98472E-06 1.01122E-05 1.02357E-05 1.03116E-05 1.03711E-05 1.04877E-05 1.06376E-05 1.0765E-05 1.09963E-05 1.11933E-05 1.15043E-05 1.17016E-05 1.19871E-05 1.224E-05 1.27725E-05 1.33052E-05 1.36959E-05 1.40057E-05 1.43886E-05 1.50466E-05 1.55746E-05 1.67538E-05 1.71916E-05 1.76614E-05 1.82025E-05 1.85369E-05 1.885E-05 1.95297E-05 2.00343E-05 2.06699E-05 2.09703E-05 2.10666E-05 2.1079E-05 2.10616E-05 2.09901E-05 2.07486E-05 2.04782E-05 2.02757E-05 2.00445E-05 1.97759E-05 1.91499E-05 1.88322E-05 1.84739E-05 1.82998E-05 1.81379E-05];",
        "k": "Tk=[314.9884 338.4416 355.4366 368.0129 378.635 391.381 402.088 409.311 419.168 427.325 437.947 449.419 459.021 467.774 472.872 479.33 490.632 500.573 514.679 531.079 546.799 560.14 572.971 588.266 610.869 629.478 644.688 658.963 673.579 690.658 705.613 714.875 731.275 743.681 757.447 765.519 774.356 787.697 803.077 817.097 825.934 837.065 847.602 858.988 872.074 878.616 890.512 901.303 908.78 915.238 921.271 925.859 934.441 958.572 977.775 984.488 988.821 996.808 1005.9 1014.822 1027.142 1034.534 1041.416 1045.239 1057.899 1063.252 1068.265 1077.781 1084.578 1091.97];\n"
              "k_t=[4.60737 4.56673 4.53764 4.51686 4.49885 4.4776 4.45959 4.44759 4.42958 4.41711 4.40002 4.38017 4.36262 4.35061 4.34045 4.33214 4.31736 4.30628 4.28827 4.26934 4.25225 4.23517 4.22039 4.19961 4.17513 4.15481 4.1368 4.12387 4.10725 4.0897 4.074 4.06477 4.04999 4.03521 4.02274 4.01443 4.0075 3.99827 3.98811 3.97703 3.97287 3.96271 3.95717 3.9484 3.93731 3.93824 3.93316 3.929 3.929 3.92715 3.92484 3.92577 3.92577 3.92577 3.92808 3.92992 3.929 3.92992 3.93316 3.935 3.94378 3.94978 3.95394 3.95809 3.9701 3.97472 3.97749 3.98719 3.99642 4.00473];"
    },
    {
        "name": "Si80Ge20B5 (P-Type 1123K)",
        "type": "P",
        "Tmax": 1123,
        "S": "Ts=[331.7216 356.8401 373.0565 396.058 419.062 440.566 457.927 476.524 500.316 533.624 557.946 601.036 617.778 632.229 648.53 661.748 683.072 700.872 712.944 735.414 758.852 783.965 805.818 826.084 844.149 859.658 889.617 912.175 934.82 983.987 1001.169 1038.439 1060.026 1080.995 1106.72 1126.807 1147.246 1161.341 1189.354 1208.82 1225.204 1234.541 1244.229 1249.778 1258.585 1267.569 1275.06];\n"
             "s_t=[0.000126197 0.000131989 0.000135532 0.000139931 0.000145601 0.000150244 0.000153641 0.000157869 0.000161486 0.000166079 0.000170258 0.000176684 0.000178834 0.000180544 0.000182572 0.000184723 0.000187508 0.000190074 0.000191711 0.000194692 0.000197599 0.000201044 0.00020405 0.000206468 0.000209058 0.00021138 0.000215118 0.000218246 0.000221104 0.000226723 0.000228727 0.000232318 0.000234565 0.000236104 0.000237642 0.000238765 0.00023935 0.000239765 0.000239714 0.000239224 0.000238856 0.000238293 0.000237754 0.000237338 0.000236531 0.00023587 0.000235552];",
        "rho": "Tr=[320.5079 333.0395 360.9164 370.8163 390.341 409.142 430.21 451.188 460.361 482.152 512.926 534.71 547.872 559.856 577.829 588.991 606.235 614.677 632.374 641.269 651.252 678.568 696.716 712.233 734.01 750.069 765.949 781.192 804.693 829.373 843.257 858.953 871.293 886.358 903.236 919.298 931.366 941.164 956.951 974.915 994.875 1003.312 1016.921 1027.625 1052.751 1063.999 1085.412 1107.454 1137.663 1148.367 1162.789 1177.03 1191.271 1213.675 1229.275 1243.876];\n"
               "r_t=[1.13328E-05 1.1532E-05 1.19831E-05 1.21811E-05 1.25602E-05 1.29835E-05 1.34239E-05 1.39277E-05 1.41702E-05 1.46626E-05 1.52932E-05 1.57113E-05 1.60031E-05 1.63189E-05 1.67241E-05 1.69302E-05 1.72841E-05 1.74922E-05 1.78703E-05 1.80896E-05 1.8321E-05 1.89209E-05 1.92869E-05 1.96105E-05 2.00434E-05 2.03117E-05 2.07003E-05 2.09865E-05 2.1556E-05 2.2109E-05 2.24954E-05 2.28543E-05 2.31614E-05 2.36902E-05 2.42494E-05 2.48357E-05 2.51614E-05 2.54002E-05 2.58655E-05 2.63344E-05 2.69199E-05 2.71572E-05 2.7577E-05 2.7735E-05 2.81887E-05 2.8338E-05 2.92258E-05 2.95743E-05 3.04694E-05 3.06532E-05 3.09424E-05 3.12276E-05 3.15082E-05 3.18938E-05 3.20952E-05 3.20953E-05];",
        "k": "Tk=[321.0237 333.2147 355.6253 371.7609 393.455 416.493 438.097 454.95 468.038 495.38 514.563 528.817 549.973 567.454 590.671 614.338 641.769 661.76 681.572 700.038 714.83 744.861 769.334 801.248 824.107 850.642 879.328 902.277 931.769 955.344 993.799 1028.579 1045.699 1071.245 1092.846 1113.102 1133.09 1156.393 1176.02 1192.78 1211.6 1228.718 1246.73 1261.517 1272.091];\n"
              "k_t=[4.86872 4.86452 4.84233 4.82734 4.79437 4.77278 4.74879 4.72421 4.71461 4.67924 4.65645 4.63966 4.60669 4.58211 4.55333 4.51616 4.47419 4.44661 4.41603 4.38965 4.36687 4.31531 4.27934 4.23677 4.1984 4.16003 4.13064 4.10486 4.08327 4.06947 4.06465 4.06463 4.07241 4.09098 4.12093 4.15148 4.18683 4.24195 4.30067 4.34561 4.41691 4.47504 4.56372 4.62784 4.68656];"
    },

    # ---------------- GeSbBiTeSe (800K, P) ----------------
    {
        "name": "Ge0.92Sb0.04Bi0.04Te0.95Se0.05 (P, 800K)",
        "type": "P",
        "Tmax": 800,
        "S": "Ts=[299.707 349.646 399.932 449.881 500.004 550.287 600.226 649.991 700.106 750.213 799.97];\n"
             "s_t=[7.56524E-05 8.79759E-05 0.000102546 0.000127706 0.000155969 0.000165191 0.0001773 0.000187698 0.000203766 0.000209243 0.000206483];",
        "rho": "Tr=[300.441 350.482 400.523 450.042 499.903 550.301 600.013 649.889 699.935 749.637 800.381];\n"
               "r_t=[4.57252E-06 5.16761E-06 5.92909E-06 7.30786E-06 1.02684E-05 1.03708E-05 9.05846E-06 9.51312E-06 1.1125E-05 1.22015E-05 1.26046E-05];",
        "k": "Tk=[299.097 347.923 398.861 450.485 499.659 550.266 600.152 651.8 702.044 752.999 800.446];\n"
             "k_t=[2.04995 1.90587 1.79976 1.5294 1.34743 1.44348 1.36261 1.32601 1.30201 1.36647 1.48771];"
    },

    # ---------------- CdAgSb (600K, P) ----------------
    {
        "name": "Cd0.99Ag0.01Sb (P, 600K)",
        "type": "P",
        "Tmax": 600,
        "S": "Ts=[298.487 320.645 340.062 362.85 385.065 403.741 426.016 448.064 473.025 497.245 517.695 535.689 556.997 574.992 598.872];\n"
             "s_t=1e-6*[198.92 212.598 222.677 234.151 247.334 254.173 260.202 263.532 266.322 264.837 263.037 258.538 254.308 248.819 240.18];",
        "rho": "Tr=[299.68 322.623 342.77 365.314 385.631 406.977 427.751 450.924 475.695 499.668 520.274 538.769 559.834 577.988 601.964];\n"
               "r_t=1e-5*[2.8206 2.91708 3.00547 3.10736 3.20925 3.28076 3.38197 3.44537 3.4865 3.47024 3.40134 3.29194 3.12245 2.97119 2.76524];",
        "k": "Tk=[302.341 326.775 352.008 373.587 400.818 425.652 450.143 475.547 501.009 524.186 548.392 574.881 603.197];\n"
             "k_t=[1.15122 1.07656 0.992119 0.941727 0.879009 0.839112 0.81841 0.797343 0.787504 0.809901 0.852577 0.922771 1.04584];"
    },

    # ---------------- MgYSbBi (600K, N) ----------------
    {
        "name": "Mg3.05Y0.012SbBi (N, 600K)",
        "type": "N",
        "Tmax": 600,
        "S": "Ts=[300 325 350 375 400 425 450 475 500 525 550 575 600];\n"
             "s_t=-[0.00018 0.00019 0.0002 0.000215 0.000225 0.000235 0.000245 0.00025 0.00026 0.00027 0.000275 0.00028 0.00029];",
        "rho": "Tr=[298.559 324.129 345.305 367.793 390.508 409.913 432 453.801 476.573 499.003 518.691 538.666 560.18 578.556 600.012];\n"
               "r_t=1e-5*[1.74657 1.76272 1.76739 1.8092 1.88678 1.98259 2.08313 2.20594 2.32403 2.41376 2.59531 2.73973 2.9206 3.08325 3.34918];",
        "k": "Tk=[301.713 327.231 348.411 372.674 398.363 422.341 449.686 476.175 503.406 525.043 548.506 575.395 603.482];\n"
             "k_t=[1.29464 1.2312 1.17176 1.13584 1.08943 1.04265 0.998764 0.962122 0.933084 0.906232 0.893138 0.901403 0.908578];"
    },

    # ---------------- CuPbSbTeSe (N, 723K) ----------------
    {
        "name": "Cu3.3Pb100Sb3Te00Se6 (N, 723K)",
        "type": "N",
        "Tmax": 723,
        "S": "Ts=[299.742 323.827 372.559 422.157 522.934 572.023 622.542 673.164 722.412];\n"
             "s_t=[-0.00018096 -0.000185941 -0.000193047 -0.000191774 -0.000186499 -0.000188702 -0.000201706 -0.00022427 -0.000264773];",
        "rho": "Tr=[298.603 323.19 372.263 423.175 472.302 522.704 571.624 622.48 672.365 722.557];\n"
               "r_t=1./[59923 56883.9 53514.7 52498.8 53247.9 54120.1 50942.5 42620.2 32423.6 23444.6];",
        "k": "Tk=[297.378 322.098 371.426 421.837 472.138 523.089 572.739 672.098 723.214];\n"
             "k_t=[1.19802 1.15854 1.10249 1.06645 1.03857 1.01254 0.990192 0.902317 0.857853];"
    },

    # ---------------- MgAgSb (600K, P) ----------------
    {
        "name": "MgAg0.97Sb0.92 (P, 600K)",
        "type": "P",
        "Tmax": 600,
        "S": "Ts=[303.2138 323.1763 347.9943 372.9239 398.155 423.08 448.041 473.189 498.184 523.289 548.317];\n"
             "s_t=[0.000174107 0.000184617 0.000194841 0.00020202 0.000206271 0.000207911 0.000206304 0.000201856 0.000194363 0.000183564 0.000169952];",
        "rho": "Tr=[266.8227 286.7758 312.0394 337.0371 362.148 387.373 412.218 437.215 462.325 487.245 512.659];\n"
               "r_t=[1.08659E-05 1.20764E-05 1.33164E-05 1.42959E-05 1.49135E-05 1.53139E-05 1.54394E-05 1.52247E-05 1.4706E-05 1.39123E-05 1.29595E-05];",
        "k": "Tk=[266.4579 286.784 311.6612 336.9557 361.605 386.255 411.514 436.887 461.466 486.615 511.841];\n"
             "k_t=[1.31396 1.2622 1.20815 1.15785 1.10293 1.059 1.03968 1.05336 1.09714 1.17421 1.2643];"
    },

    # ---------------- YbCoSb (873K, N) ----------------
    {
        "name": "Yb0.3Co4Sb12 (N, 873K)",
        "type": "N",
        "Tmax": 873,
        "S": "Ts=[300.1985 350.2942 400.192 450.782 500.184 550.18 600.472 650.565 700.163 750.453 800.147 850.333];\n"
             "s_t=[-0.000114219 -0.000123461 -0.000132704 -0.000141981 -0.000149559 -0.000157526 -0.000164501 -0.000170449 -0.000175759 -0.0001799 -0.000181952 -0.000178441];",
        "rho": "Tr=[300.6001 350.3174 400.133 450.245 500.75 550.468 600.285 650.397 700.608 750.031 799.947 850.552];\n"
               "r_t=[3.59052E-06 3.89397E-06 4.18476E-06 4.44179E-06 4.72414E-06 4.98961E-06 5.22555E-06 5.41929E-06 5.62991E-06 5.82365E-06 6.00052E-06 6.17316E-06];",
        "k": "Tk=[300.1218 350.41 400.599 450.69 499.891 550.179 600.171 650.064 699.957 749.85 800.04 850.328];\n"
             "k_t=[3.88613 3.74168 3.59311 3.52123 3.45182 3.42365 3.40042 3.42172 3.44633 3.50886 3.57469 3.67846];"
    },

    # ---------------- SnGeSeS (873K, P) ----------------
    {
        "name": "Sn0.96Ge0.04Se0.96S0.04 (P, 873K)",
        "type": "P",
        "Tmax": 873,
        "S": "Ts=[299.749 321.905 372.187 423.054 473.258 522.509 572.647 622.779 673.119 722.762 772.814 822.511 872.934];\n"
             "s_t=[3.06E-04 3.13E-04 3.34E-04 3.54E-04 3.72E-04 3.93E-04 4.19E-04 4.38E-04 4.46E-04 4.22E-04 3.69E-04 3.29E-04 3.45E-04];",
        "rho": "Tr=[299.367 322.702 373.648 422.239 472.217 522.106 572.644 623.03 673.203 723.179 773.179 822.821 873.449];\n"
               "r_t=1./[900.724 1178.77 1988.39 2752.21 3224.05 3205.98 2769.96 2190 1876.67 2247.09 3393.68 4801.99 4883.7];",
        "k": "Tk=[299.915 323.739 373.399 423.573 473.44 523.616 574.294 623.264 673.032 723.201 772.975 823.057 873.332];\n"
             "k_t=[1.02248 0.905408 0.749523 0.699682 0.596819 0.565005 0.539555 0.477625 0.383245 0.288441 0.244751 0.258112 0.213574];"
    },

    # ---------------- MgSbBiMnTe (573K, N) ----------------
    {
        "name": "Mg3.2Sb0.5Bi1.495Mn0.02Te0.005 (N, 573K)",
        "type": "N",
        "Tmax": 573,
        "S": "Ts=[299.502 322.257 347.838 372.536 397.411 422.697 472.674 523.293 573.203];\n"
             "s_t=[-1.72E-04 -1.76E-04 -1.84E-04 -1.91E-04 -1.98E-04 -2.04E-04 -2.11E-04 -2.13E-04 -2.12E-04];",
        "rho": "Tr=[299.589 322.91 348.404 372.841 397.572 422.949 472.822 523.047 572.626];\n"
               "r_t=1./[1.11E+05 1.03E+05 94448.1 86827 79978.6 74072.6 63409.9 54914.4 48001.5];",
        "k": "Tk=[300.156 323.461 348.059 373.425 398.444 423.646 472.658 523.751 574.1];\n"
             "k_t=[1.2634 1.22067 1.17865 1.14375 1.11814 1.10284 1.10442 1.1486 1.22496];"
    },
      # ---------------- Ag0.98Mn0.02CuTe (773K, P) ----------------
    {
    "name": "Ag0.98Mn0.02CuTe",
    "type": "P",
    "Tmax": 773.0,
    "S": "Ts=[303\t323\t373\t423\t473\t523\t573\t623\t673\t723\t773];\ns_t=[6.37342E-05\t6.53623E-05\t7.27898E-05\t0.000116998\t0.000190448\t0.000228663\t0.000241559\t0.000248973\t0.000250093\t0.00025221\t0.000252317];",
    "k": "Tk=[303\t323\t373\t423\t473\t523\t573\t623\t673\t723\t773];\nk_t=[0.78903\t0.78083\t0.7173\t0.49596\t0.36685\t0.33611\t0.35045\t0.3689\t0.37095\t0.37914\t0.38734];",
    "rho": "Tr=[303\t323\t373\t423\t473\t523\t573\t623\t673\t723\t773];\nr_t=1./[78673.5\t76938.22\t62437.56\t29893.73\t13662.79\t13427.83\t13293.48\t13492.67\t13730.89\t13932.22\t14793.83];"
    },
     # ---------------- SnSe-0.75%Te-0.7%Mo (723K, N) ----------------
    {
    "name": "SnSe-0.75%Te-0.7%Mo (N-Type 723K)",
    "type": "N",
    "Tmax": 723.0,
    "S": "Ts=[300\t323\t373\t423\t473\t523\t573\t623\t673\t723];\ns_t=[-0.000224\t-0.00023\t-0.000242\t-0.000256\t-0.000272\t-0.000291\t-0.000312\t-0.000334\t-0.000354\t-0.000375];",
    "k": "Tk=[300\t323\t373\t423\t473\t523\t573\t623\t673\t723];\nk_t=[1.4\t1.34\t1.21\t1.09\t0.98\t0.89\t0.82\t0.76\t0.7\t0.63];",
    "rho": "Tr=[300\t323\t373\t423\t473\t523\t573\t623\t673\t723];\nr_t=1./[54500\t49800\t40200\t31800\t24800\t19200\t14800\t11000\t8200\t6000];"
    },
    # ---------------- Skutterudite (P, 873K) ----------------
    {
        "name": "Skutterudite Ce0.9Fe3.5Co0.5Sb12 (P, 873K)",
        "type": "P",
        "Tmax": 873,
        "S": "Ts=[322.817 347.926 373.034 398.056 423.078 447.929 472.949 497.714 523.07 547.835 573.106 597.785 622.971 647.818 672.919 698.02 722.951 747.798 772.981 797.826 822.67 847.937 872.611];\n"
             "s_t=[0.000080459 9.00436E-05 9.85632E-05 0.000106532 0.000113656 0.000119569 0.00012482 0.0001293 0.000133339 0.000136828 0.00013995 0.000142887 0.000145495 0.000148065 0.000150379 0.000152252 0.000153941 0.000154969 0.000155593 0.00015552 0.000154712 0.0001535 0.000151848];",
        "rho": "Tr=[298.127 323.037 348.244 372.857 398.36 423.072 448.18 472.793 498.099 523.107 547.919 573.224 598.134 623.044 647.855 673.061 697.971 723.078 748.087 773.095 798.301 822.913 848.119 872.929];\n"
               "r_t=[6.14388E-06 6.30307E-06 6.45191E-06 6.58348E-06 6.72197E-06 6.83627E-06 6.97821E-06 7.12014E-06 7.24826E-06 7.39365E-06 7.54939E-06 7.67751E-06 7.81944E-06 7.95102E-06 8.08949E-06 8.21416E-06 8.32502E-06 8.44624E-06 8.56055E-06 8.66795E-06 8.74083E-06 8.83442E-06 8.89349E-06 8.93184E-06];",
        "k": "Tk=[297.742 323.05 347.971 373.085 398.102 422.829 448.04 472.96 498.075 522.898 547.818 572.933 597.756 622.773 647.984 672.905 697.922 723.036 748.054 772.974 798.088 823.009 847.929 872.946];\n"
             "k_t=[2.09336 2.08952 2.09355 2.09364 2.10159 2.10954 2.11488 2.12938 2.13733 2.14659 2.15454 2.16512 2.177 2.18757 2.20207 2.22574 2.25334 2.28879 2.34128 2.40162 2.47637 2.56553 2.6704 2.79099];"
    },

    # ---------------- Skutterudite (N, 873K) ----------------
    {
        "name": "Skutterudite Ba0.05Yb0.15Co4Sb12 (N, 873K)",
        "type": "N",
        "Tmax": 873,
        "S": "Ts=[322.839 347.963 372.832 397.533 422.91 448.118 472.818 497.942 522.727 547.85 572.889 597.843 622.882 647.92 672.875 697.998 722.952 747.906 773.03 797.815 822.938 848.061 873.1];\n"
             "s_t=[-0.000127107 -0.000141542 -0.000152495 -0.000161285 -0.000168221 -0.000173894 -0.000178921 -0.000183218 -0.000187065 -0.000190407 -0.00019344 -0.000196024 -0.000198439 -0.000200405 -0.00020209 -0.000203466 -0.000204589 -0.000205376 -0.000206134 -0.000206527 -0.000206696 -0.000206667 -0.00020605];",
        "rho": "Tr=[297.567 322.597 347.726 372.854 397.587 422.913 447.844 472.873 497.804 522.932 548.06 572.891 597.92 622.851 647.88 673.107 697.839 723.066 748.095 773.025 798.152 823.083 848.21 873.14];\n"
               "r_t=[5.3536E-06 5.62292E-06 5.89224E-06 6.04494E-06 6.50867E-06 6.66135E-06 6.89181E-06 7.16114E-06 7.46935E-06 7.58318E-06 7.8525E-06 7.92749E-06 8.11907E-06 8.42728E-06 8.46338E-06 8.69381E-06 8.80768E-06 8.99924E-06 9.11308E-06 9.22693E-06 9.26301E-06 9.49348E-06 9.49069E-06 9.72116E-06];",
        "k": "Tk=[298.31 323.13 348.144 373.158 397.978 423.089 447.812 472.922 498.033 522.95 547.964 573.172 597.992 622.909 647.922 672.936 697.95 723.158 747.881 773.089 797.812 822.922 848.13 873.047];\n"
             "k_t=[2.66099 2.61952 2.58939 2.56756 2.55272 2.5453 2.54268 2.54486 2.54879 2.55534 2.56276 2.57498 2.58895 2.60685 2.6313 2.66492 2.70552 2.75573 2.81467 2.88103 2.95263 3.02423 3.08928 3.13774];"
    }
] 

# ============================ Data Parsing & Units ============================

def extract_data_from_text(txt, varname):
    txt = txt.replace('\n', '').replace(';', '')
    match = re.search(r'%s\s*=\s*([-\d\.eE]+)\s*\*\s*\[([^\]]+)\]' % varname, txt)
    if match:
        factor = float(match.group(1))
        arr = [float(x.replace('E','e')) for x in match.group(2).replace(',', ' ').split()]
        return np.array(arr) * factor
    match = re.search(r'%s\s*=\s*1\./\s*\[([^\]]+)\]' % varname, txt)
    if match:
        arr = [float(x.replace('E','e')) for x in match.group(1).replace(',', ' ').split()]
        return 1.0 / np.array(arr)
    match = re.search(r'%s\s*=\s*-\s*\[([^\]]+)\]' % varname, txt)
    if match:
        arr = [float(x.replace('E','e')) for x in match.group(1).replace(',', ' ').split()]
        return -np.array(arr)
    match = re.search(r'%s\s*=\s*\[([^\]]+)\]' % varname, txt)
    if match:
        arr = [float(x.replace('E','e')) for x in match.group(1).replace(',', ' ').split()]
        return np.array(arr)
    return None

def parse_input(txt, varname, deg=4):
    txt = txt.strip()
    if varname == 'S':
        arrx, arry = extract_data_from_text(txt, 'Ts'), extract_data_from_text(txt, 's_t')
    elif varname == 'k':
        arrx, arry = extract_data_from_text(txt, 'Tk'), extract_data_from_text(txt, 'k_t')
    elif varname == 'rho':
        arrx, arry = extract_data_from_text(txt, 'Tr'), extract_data_from_text(txt, 'r_t')
    else:
        arrx = arry = None
    if arrx is not None and arry is not None:
        return np.polyfit(arrx, arry, deg)
    if txt.startswith('[') and txt.endswith(']'):
        return np.array([float(x) for x in txt[1:-1].replace(';','').replace(',', ' ').split()])
    if ('T' in txt or 't' in txt) and ('=' not in txt):
        expr = txt.replace('^','**')
        xs = np.linspace(300, 900, 13)
        ys = np.array([eval(expr, {'T':x, 't':x, 'np':np}) for x in xs])
        return np.polyfit(xs, ys, deg)
    raise ValueError(f"Failed to parse {varname} input")

def unit_convert_temp(val, unit):
    if unit == "K": return float(val)
    if unit == "°C": return float(val) + 273.15
    raise ValueError("Unsupported temperature unit")

def unit_convert_length(val, unit):
    if unit == "m": return float(val)
    if unit == "mm": return float(val) / 1000
    raise ValueError("Unsupported length unit")

def unit_convert_resist(val, unit):
    v = float(val)
    if unit == "Ω·m²": return v
    if unit == "Ω·cm²": return v * 1e-4
    if unit == "Ω·mm²": return v * 1e-6
    raise ValueError("Unsupported contact resistivity unit")

def safe_polyval_k(p, T, k_min=1e-5):
    return np.maximum(np.polyval(p, T), k_min)

def safe_polyint_k(p, T1, T2, k_min=1e-5):
    result, _ = quad(lambda T: safe_polyval_k(p, T, k_min), T1, T2, limit=200)
    return result

# ============================ Core Computation ============================

def run_calc_single(p_st, p_kt, p_rt, Qin, Tc, Tmax, L, gamma_c_h, gamma_c_c):
    Sum_st = np.polyint(p_st)
    Sum_rt = np.polyint(p_rt)
    p_taut = np.convolve(np.polyder(p_st), [1, 0])
    Sum_taut = np.polyint(p_taut)
    deltaT = Tmax - Tc
    effc = deltaT / Tmax
    SengP = np.polyval(Sum_st, Tmax) - np.polyval(Sum_st, Tc)
    KengP = safe_polyint_k(p_kt, Tc, Tmax)
    RengP = np.polyval(Sum_rt, Tmax) - np.polyval(Sum_rt, Tc)
    RengP1 = RengP + deltaT * (gamma_c_h + gamma_c_c) / L
    ZT_eng = SengP ** 2 * deltaT / KengP / RengP1

    int_taut, _ = quad(lambda T: np.polyval(Sum_taut, Tmax) - np.polyval(Sum_taut, T), Tc, Tmax, limit=200)
    int_rt, _ = quad(lambda T: np.polyval(Sum_rt, Tmax) - np.polyval(Sum_rt, T), Tc, Tmax, limit=200)

    a0 = np.polyval(p_st, Tmax) * deltaT / SengP - int_taut / (SengP * deltaT) * effc
    a1 = a0 - effc * (int_rt + deltaT ** 2 * gamma_c_h / L) / RengP1 / deltaT
    a2 = a0 - 2 * effc * (int_rt + deltaT ** 2 * gamma_c_h / L) / RengP1 / deltaT

    m = np.sqrt(1 + ZT_eng * a1 / effc)
    eff = effc * (np.sqrt(1 + ZT_eng * a1 / effc) - 1) / (a0 * np.sqrt(1 + ZT_eng * a1 / effc) + a2)

    den = (
        KengP
        + 1/(1+m) * (deltaT * SengP * Tmax * np.polyval(p_st, Tmax) - SengP * int_taut) / RengP1
        - 1/(1+m)**2 * SengP**2 * (int_rt + deltaT**2 * gamma_c_h / L) / (RengP1**2)
    )
    alpha = Qin / den
    Vopt = abs(SengP * m / (1 + m))

    R_star_opt = (1 / alpha) * (1 / deltaT) * RengP1 if (alpha != 0 and deltaT != 0) else 0
    R_L_opt = m * R_star_opt

    def sc_eq(Th_sc):
        dt = Th_sc - Tc
        if dt <= 0: return Qin
        SengP_sc = np.polyval(Sum_st, Th_sc) - np.polyval(Sum_st, Tc)
        KengP_sc = safe_polyint_k(p_kt, Tc, Th_sc)
        RengP_sc = np.polyval(Sum_rt, Th_sc) - np.polyval(Sum_rt, Tc)
        RengP1_sc = RengP_sc + dt * (gamma_c_h + gamma_c_c) / L
        int_taut_sc, _ = quad(lambda T: np.polyval(Sum_taut, Th_sc) - np.polyval(Sum_taut, T), Tc, Th_sc, limit=200)
        int_rt_sc, _ = quad(lambda T: np.polyval(Sum_rt, Th_sc) - np.polyval(Sum_rt, T), Tc, Th_sc, limit=200)
        den_sc = (
            KengP_sc
            + (dt * SengP_sc * Th_sc * np.polyval(p_st, Th_sc) - SengP_sc * int_taut_sc) / RengP1_sc
            - SengP_sc**2 * (int_rt_sc + dt**2 * gamma_c_h / L) / (RengP1_sc**2)
        )
        return alpha * den_sc - Qin

    Th_sc = fsolve(sc_eq, Tc + 20)[0]
    dt_sc = Th_sc - Tc
    SengP_sc = np.polyval(Sum_st, Th_sc) - np.polyval(Sum_st, Tc)
    RengP_sc = np.polyval(Sum_rt, Th_sc) - np.polyval(Sum_rt, Tc)
    RengP1_sc = RengP_sc + dt_sc * (gamma_c_h + gamma_c_c) / L
    R_star_sc = (1 / alpha) * (1 / dt_sc) * RengP1_sc if (alpha != 0 and dt_sc != 0) else 0
    Isc = abs(SengP_sc / R_star_sc) if R_star_sc != 0 else 0

    return eff, m, alpha, Vopt, Th_sc, Isc, R_star_opt, R_L_opt

def run_calc_Couple(Qin, Tc, Th, L, rc_nh, rc_nc, rc_ph, rc_pc,
                  p_st, p_kt, p_rt, n_st, n_kt, n_rt):
    Sum_pst, Sum_prt = np.polyint(p_st), np.polyint(p_rt)
    p_taut = np.convolve(np.polyder(p_st), [1, 0]); Sum_ptaut = np.polyint(p_taut)
    Sum_nst, Sum_nrt = np.polyint(n_st), np.polyint(n_rt)
    n_taut = np.convolve(np.polyder(n_st), [1, 0]); Sum_ntaut = np.polyint(n_taut)

    deltaT = Th - Tc
    effc   = deltaT / Th

    SengP  = np.polyval(Sum_pst, Th) - np.polyval(Sum_pst, Tc)
    KengP  = safe_polyint_k(p_kt, Tc, Th)
    RengP  = np.polyval(Sum_prt, Th) - np.polyval(Sum_prt, Tc)

    SengN  = np.polyval(Sum_nst, Th) - np.polyval(Sum_nst, Tc)
    KengN  = safe_polyint_k(n_kt, Tc, Th)
    RengN  = np.polyval(Sum_nrt, Th) - np.polyval(Sum_nrt, Tc)

    RengP1 = RengP + deltaT * (rc_ph + rc_pc) / L
    RengN1 = RengN + deltaT * (rc_nh + rc_nc) / L

    beta = np.sqrt(RengN1 * KengP / (RengP1 * KengN)) if RengP1 * KengN != 0 else 1.0
    ZT_eng = (SengP - SengN) ** 2 * deltaT / (np.sqrt(KengP * RengP1) + np.sqrt(KengN * RengN1)) ** 2

    int_ptaut, _ = quad(lambda T: np.polyval(Sum_ptaut, Th) - np.polyval(Sum_ptaut, T), Tc, Th, limit=200)
    int_ntaut, _ = quad(lambda T: np.polyval(Sum_ntaut, Th) - np.polyval(Sum_ntaut, T), Tc, Th, limit=200)
    int_prt,  _  = quad(lambda T: np.polyval(Sum_prt , Th) - np.polyval(Sum_prt , T), Tc, Th, limit=200)
    int_nrt,  _  = quad(lambda T: np.polyval(Sum_nrt , Th) - np.polyval(Sum_nrt , T), Tc, Th, limit=200)

    a0 = (np.polyval(p_st, Th) - np.polyval(n_st, Th)) * deltaT / (SengP - SengN) \
         - (int_ptaut - int_ntaut) / ((SengP - SengN) * deltaT) * effc
    a1 = a0 - effc * (int_prt + beta * int_nrt + deltaT ** 2 * (rc_ph + beta * rc_nh) / L) \
              / (RengP1 + beta * RengN1) / deltaT
    a2 = a0 - 2 * effc * (int_prt + beta * int_nrt + deltaT ** 2 * (rc_ph + beta * rc_nh) / L) \
              / (RengP1 + beta * RengN1) / deltaT

    m    = np.sqrt(1 + ZT_eng * a1 / effc)
    eff = effc * (np.sqrt(1 + ZT_eng * a1 / effc) - 1) \
        / (a0 * np.sqrt(1 + ZT_eng * a1 / effc) + a2)

    denP = KengP \
         + ((SengP-SengN)* Th * np.polyval(p_st, Th) * deltaT -(SengP-SengN) * int_ptaut) / (1 + m) / ( RengP1+beta ** -1*RengN1 ) \
         - (SengP-SengN) ** 2 * (int_prt + deltaT ** 2 * rc_ph / L) / (1 + m) ** 2 / ( RengP1+beta ** -1*RengN1 ) ** 2

    denN = KengN \
         - ((SengP-SengN)* Th * np.polyval(n_st, Th) * deltaT -(SengP-SengN) * int_ntaut) / (1 + m) /( beta *RengP1+RengN1 ) \
         - (SengP-SengN)** 2 * (int_nrt + deltaT ** 2 * rc_nh / L) / (1 + m) ** 2 / (beta * RengP1+RengN1 ) ** 2

    alphaP    = Qin / (denP + beta * denN)
    alphaN   = beta * alphaP
    alpha   = alphaN + alphaP
    Pmax     = Qin * eff
    Vopt     = (SengP - SengN) * m / (1 + m)

    if deltaT == 0 or alphaP == 0 or alphaN == 0:
        R_star_opt = 0.0
    else:
        R_star_opt = (RengP1 / alphaP + RengN1 / alphaN) / deltaT
    R_L_opt = m * R_star_opt

    def sc_eq(Th_sc):
        dt = Th_sc - Tc
        if dt <= 0: return Qin
        KengP_sc = safe_polyint_k(p_kt, Tc, Th_sc)
        KengN_sc = safe_polyint_k(n_kt, Tc, Th_sc)
        SengP_sc = np.polyval(Sum_pst, Th_sc) - np.polyval(Sum_pst, Tc)
        SengN_sc = np.polyval(Sum_nst, Th_sc) - np.polyval(Sum_nst, Tc)
        RengP_sc = np.polyval(Sum_prt, Th_sc) - np.polyval(Sum_prt, Tc)
        RengN_sc = np.polyval(Sum_nrt, Th_sc) - np.polyval(Sum_nrt, Tc)
        RengP1_sc = RengP_sc + dt * (rc_ph + rc_pc) / L
        RengN1_sc = RengN_sc + dt * (rc_nh + rc_nc) / L

        int_ptaut_sc, _ = quad(lambda T: np.polyval(Sum_ptaut, Th_sc) - np.polyval(Sum_ptaut, T), Tc, Th_sc, limit=200)
        int_ntaut_sc, _ = quad(lambda T: np.polyval(Sum_ntaut, Th_sc) - np.polyval(Sum_ntaut, T), Tc, Th_sc, limit=200)
        int_prt_sc,  _  = quad(lambda T: np.polyval(Sum_prt , Th_sc) - np.polyval(Sum_prt , T), Tc, Th_sc, limit=200)
        int_nrt_sc,  _  = quad(lambda T: np.polyval(Sum_nrt , Th_sc) - np.polyval(Sum_nrt , T), Tc, Th_sc, limit=200)

        denP_sc = KengP_sc \
                + ((SengP_sc - SengN_sc) * Th_sc * np.polyval(p_st, Th_sc) * dt -(SengP_sc-SengN_sc) * int_ptaut_sc) /( RengP1_sc+beta ** -1*RengN1_sc ) \
                - (SengP_sc - SengN_sc) ** 2 * (int_prt_sc + dt ** 2 * rc_ph / L) / ( RengP1_sc+beta ** -1*RengN1_sc ) ** 2
        denN_sc = KengN_sc \
                - ((SengP_sc-SengN_sc) * Th_sc * np.polyval(n_st, Th_sc) * dt -(SengP_sc-SengN_sc) * int_ntaut_sc) /( beta*RengP1_sc+RengN1_sc ) \
                - (SengP-SengN_sc)** 2 * (int_nrt_sc + dt ** 2 * rc_nh / L) / (beta* RengP1_sc+RengN1_sc )  ** 2

        return alphaP * (denP_sc + beta * denN_sc) - Qin

    Th_sc = fsolve(sc_eq, Tc + 20)[0]
    dt_sc = Th_sc - Tc
    SengP_sc  = np.polyval(Sum_pst, Th_sc) - np.polyval(Sum_pst, Tc)
    SengN_sc  = np.polyval(Sum_nst, Th_sc) - np.polyval(Sum_nst, Tc)
    RengP_sc  = np.polyval(Sum_prt, Th_sc) - np.polyval(Sum_prt, Tc)
    RengN_sc  = np.polyval(Sum_nrt, Th_sc) - np.polyval(Sum_nrt, Tc)
    RengP1_sc = RengP_sc + dt_sc * (rc_ph + rc_pc) / L
    RengN1_sc = RengN_sc + dt_sc * (rc_nh + rc_nc) / L

    if dt_sc == 0 or alphaP == 0 or alphaN == 0:
        R_star_sc = 0.0
    else:
        R_star_sc = (RengP1_sc / alphaP + RengN1_sc / alphaN) / dt_sc
    Isc = (SengP_sc - SengN_sc) / R_star_sc if R_star_sc != 0 else 0.0

    return eff, m, alpha, beta, alphaP, alphaN, Qin*eff, Vopt, Th_sc, Isc, R_star_opt, R_L_opt

# ============================ GUI ============================

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1260x800")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.intro = IntroFrame(self, self.show_main)
        self.intro.pack(fill="both", expand=True)
        self.main = None

    def on_closing(self):
        try:
            self.destroy()
        except:
            pass
        sys.exit(0)

    def show_main(self):
        self.intro.pack_forget()
        self.main = TEGFrame(self)
        self.main.pack(fill="both", expand=True)

class IntroFrame(tk.Frame):
    def __init__(self, parent, switch_callback):
        super().__init__(parent)
        tk.Label(self, text="Constant Heat-Flux Thermoelectric Leg/Couple Optimization Tool", font=("Microsoft YaHei", 25, "bold")).pack(pady=60)
        tk.Label(self, text="Test version, for internal research only. Please do not distribute!", font=("Microsoft YaHei", 15, "bold"), fg="red").pack(pady=12)
        tk.Label(self, text="Author: Liuyouhong", font=("Microsoft YaHei", 18)).pack(pady=10)
        tk.Button(self, text="Start Calculation", font=("Microsoft YaHei", 19, "bold"), width=16, bg="#3c87f6", fg="white",
                  command=switch_callback).pack(pady=(55, 32))
        self.note_btn = tk.Button(self, text="Notes", font=("Microsoft YaHei", 10), command=self.show_notes,
                                  width=9, height=1, bg="#f6f6f6", fg="#3c87f6", relief="flat", bd=0,
                                  activebackground="#e2e6ee", activeforeground="#0c286c", cursor="hand2")
        self.note_btn.pack(pady=(70, 10))
        self.note_btn.bind("<Enter>", lambda e: self.note_btn.config(bg="#e2e6ee"))
        self.note_btn.bind("<Leave>", lambda e: self.note_btn.config(bg="#f6f6f6"))

    def show_notes(self):
        notes = ("[Software Notes]\n\nThis program performs optimization design for thermoelectric single-leg/Couple structures under fixed heat flux.\n"
                 "- 1D heat transfer only; contact resistances supported; cold-side temperature fixed.\n"
                 "- Material parameters support: raw tables, fitting expressions, or polynomial coefficients.\n"
                 "- Non-negativity constraint for high-temperature extrapolation of k(T); open/short-circuit points are solved by the energy equation.\n"
                 "- On first run, a file \"Custom_Material_Library.txt\" (commented JSON) is created in the program directory; you can save via the UI or edit manually; it loads automatically next start.\n")
        win = tk.Toplevel(self); win.title("Notes"); win.resizable(False, False)
        tk.Label(win, text=notes, font=("Microsoft YaHei", 13), justify="left").pack(padx=24, pady=20)
        tk.Button(win, text="Close", font=("Microsoft YaHei", 13), width=10, command=win.destroy).pack(pady=(0, 16))

# ---------- Unified "Material Name + Type + Tmax" input dialog ----------
class MaterialMetaDialog(tk.Toplevel):
    def __init__(self, parent, need_type=True, default_type="P"):
        super().__init__(parent)
        self.title("Save to Custom Library")
        self.resizable(False, False)
        self.result = None
        frm = tk.Frame(self); frm.pack(padx=16, pady=14)
        tk.Label(frm, text="Material name:", font=("Microsoft YaHei", 11)).grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.e_name = tk.Entry(frm, width=32, font=("Microsoft YaHei", 11)); self.e_name.grid(row=0, column=1, sticky="w")
        self.need_type = need_type
        if need_type:
            tk.Label(frm, text="Type:", font=("Microsoft YaHei", 11)).grid(row=1, column=0, sticky="e", padx=6, pady=6)
            self.type_var = tk.StringVar(value=default_type)
            ttk.Combobox(frm, textvariable=self.type_var, values=["P","N"], width=6, state="readonly").grid(row=1, column=1, sticky="w")
            r = 2
        else:
            r = 1
        tk.Label(frm, text="Tmax (K):", font=("Microsoft YaHei", 11)).grid(row=r, column=0, sticky="e", padx=6, pady=6)
        self.e_tmax = tk.Entry(frm, width=12, font=("Microsoft YaHei", 11)); self.e_tmax.grid(row=r, column=1, sticky="w")

        btn = tk.Frame(self); btn.pack(pady=(0,12))
        tk.Button(btn, text="OK", width=10, command=self._ok).pack(side="left", padx=8)
        tk.Button(btn, text="Cancel", width=10, command=self.destroy).pack(side="left", padx=8)
        self.grab_set(); self.e_name.focus_force()

    def _ok(self):
        name = self.e_name.get().strip()
        if not name:
            messagebox.showwarning("Warning","Material name cannot be empty.", parent=self); return
        try:
            tmax = float(self.e_tmax.get().strip())
        except Exception:
            messagebox.showwarning("Warning","Invalid Tmax.", parent=self); return
        dtype = self.type_var.get() if self.need_type else None
        self.result = {"name": name, "type": dtype, "Tmax": tmax}
        self.destroy()

# ---------- Main UI ----------
class TEGFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.mode_var = tk.StringVar(value="single")
        self._input_cache = {"single": {}, "Couple": {}}
        self._last_mode = "single"
        self.last_results = None
        self.build_gui()
        self.restore_inputs("single")

    def build_gui(self):
        top = tk.Frame(self); top.pack(fill="x", padx=10, pady=(6,2))
        tk.Label(top, text="Select computation mode:", font=("Microsoft YaHei", 12)).pack(side="left")
        tk.Radiobutton(top, text="Single Leg", font=("Microsoft YaHei", 12), variable=self.mode_var, value="single",
                       command=lambda: self.switch_mode("single")).pack(side="left", padx=6)
        tk.Radiobutton(top, text="Single Couple", font=("Microsoft YaHei", 12), variable=self.mode_var, value="Couple",
                       command=lambda: self.switch_mode("Couple")).pack(side="left", padx=6)
        mat_btn = tk.Button(top, text="Material Library", font=("Microsoft YaHei", 11, "bold"), command=self.open_material_lib, bg="#F7EFE6", activebackground="#F1E7D6", fg="#5A4631", activeforeground="#5A4631", relief="ridge", bd=2, padx=14, pady=6, cursor="hand2"); mat_btn.pack(side="left", padx=(8,0)); mat_btn.bind("<Enter>", lambda e: mat_btn.config(bg="#F4E9DE")); mat_btn.bind("<Leave>", lambda e: mat_btn.config(bg="#F7EFE6"))

        self.material_frame = tk.Frame(self); self.material_frame.pack(fill="x", padx=8, pady=(0, 6))
        self.mat_frms = {}; self.init_material_ui()
        self.material_frame.winfo_children()[1].pack_forget()
        self.material_frame.winfo_children()[2].pack_forget()

        param_frame = tk.LabelFrame(self, text="Design Parameters", font=("Microsoft YaHei", 12)); param_frame.pack(fill="x", padx=10, pady=5)
        for i in range(8): param_frame.grid_columnconfigure(i, weight=1)
        tk.Label(param_frame, text="Tmax,safe (Max safe temperature)", font=("Microsoft YaHei", 11), anchor="w", width=27).grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.tmax_entry = tk.Entry(param_frame, width=12, font=("Microsoft YaHei", 12)); self.tmax_entry.grid(row=0, column=1, sticky="w", padx=2)
        self.tmax_unit_var = tk.StringVar(value="K"); ttk.Combobox(param_frame, textvariable=self.tmax_unit_var, values=["K", "°C"], width=4, state="readonly").grid(row=0, column=2, sticky="w")

        tk.Label(param_frame, text="Tc (Cold-side temperature)", font=("Microsoft YaHei", 11), anchor="w", width=22).grid(row=0, column=3, sticky="w", padx=5, pady=3)
        self.tc_entry = tk.Entry(param_frame, width=12, font=("Microsoft YaHei", 12)); self.tc_entry.grid(row=0, column=4, sticky="w", padx=2)
        self.tc_unit_var = tk.StringVar(value="K"); ttk.Combobox(param_frame, textvariable=self.tc_unit_var, values=["K", "°C"], width=4, state="readonly").grid(row=0, column=5, sticky="w")

        tk.Label(param_frame, text="Input heat flow Qin (W)", font=("Microsoft YaHei", 11), anchor="w", width=22).grid(row=0, column=6, sticky="w", padx=5, pady=3)
        self.qin_entry = tk.Entry(param_frame, width=12, font=("Microsoft YaHei", 12)); self.qin_entry.grid(row=0, column=7, sticky="w", padx=2)

        self.use_resist_var = tk.StringVar(value="No")
        tk.Label(param_frame, text="Consider contact resistances", font=("Microsoft YaHei", 11), anchor="w", width=22).grid(row=1, column=0, sticky="w", padx=5, pady=3)
        tk.Radiobutton(param_frame, text="No", variable=self.use_resist_var, value="No", font=("Microsoft YaHei", 10),
                       command=self.toggle_resist_area).grid(row=1, column=1, sticky="w")
        tk.Radiobutton(param_frame, text="Yes", variable=self.use_resist_var, value="Yes", font=("Microsoft YaHei", 10),
                       command=self.toggle_resist_area).grid(row=1, column=2, sticky="w")

        self.resist_frame = tk.Frame(param_frame); self.resist_frame.grid(row=2, column=0, columnspan=8, sticky="w", padx=0)
        self.build_resist_inputs(); self.toggle_resist_area()

        btn_frame = tk.Frame(self); btn_frame.pack(pady=(10, 6))
        tk.Button(btn_frame, text="Run Optimization", font=("Microsoft YaHei", 13, "bold"),
                  bg="#3c87f6", fg="white", width=18, command=self.on_calc).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Generate Load Curves", font=("Microsoft YaHei", 13, "bold"),
                  bg="#4CAF50", fg="white", width=18, command=self.on_generate_curves).pack(side="left", padx=10)

        result_frame = tk.LabelFrame(self, text="Optimization Results", font=("Microsoft YaHei", 12))
        result_frame.pack(fill="both", padx=14, pady=12, expand=True)
        self.result_text = tk.Text(result_frame, font=("Microsoft YaHei", 15, "bold"), height=12, wrap="none", bg="#fcfcfd")
        self.result_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.result_text.config(state="disabled")

    def init_material_ui(self):
        for w in self.material_frame.winfo_children(): w.destroy()
        frm_single = tk.LabelFrame(self.material_frame, text="Single-Leg Material Parameters (paste raw data, fitting expression, or coefficients)", font=("Microsoft YaHei", 12))
        frm_single.pack(fill="x", padx=4, pady=(3, 2))
        self.mat_frms["single"] = {name: None for name in ['S', 'k', 'rho']}
        for i, (name, label, _) in enumerate([('S', 'Seebeck coefficient S(T), V/K', ''), ('rho', 'Resistivity ρ(T), Ω·m', ''), ('k', 'Thermal conductivity κ(T), W/(m·K)', '')]):
            tk.Label(frm_single, text=f"{label}:", font=("Microsoft YaHei", 11), anchor="w").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            txt = tk.Text(frm_single, height=2, font=("Consolas", 11), wrap="none", width=140); txt.grid(row=i, column=1, sticky="ew", padx=2, pady=2); self.mat_frms["single"][name] = txt

        frm_n = tk.LabelFrame(self.material_frame, text="N-Leg Material Parameters (paste raw data, fitting expression, or coefficients)", font=("Microsoft YaHei", 12), fg="blue")
        frm_p = tk.LabelFrame(self.material_frame, text="P-Leg Material Parameters (paste raw data, fitting expression, or coefficients)", font=("Microsoft YaHei", 12), fg="red")
        self.mat_frms["n"], self.mat_frms["p"] = {}, {}
        for f, pf, _ in [(frm_n, self.mat_frms["n"], "N"), (frm_p, self.mat_frms["p"], "P")]:
            f.pack(fill="x", padx=4, pady=(3, 2))
            for i, (name, label, _) in enumerate([('S', 'Seebeck coefficient S(T), V/K', ''), ('rho', 'Resistivity ρ(T), Ω·m', ''), ('k', 'Thermal conductivity κ(T), W/(m·K)', '')]):
                tk.Label(f, text=f"{label}:", font=("Microsoft YaHei", 11), anchor="w").grid(row=i, column=0, sticky="w", padx=5, pady=2)
                txt = tk.Text(f, height=2, font=("Consolas", 11), wrap="none", width=140); txt.grid(row=i, column=1, sticky="ew", padx=2, pady=2); pf[name] = txt

    def save_inputs(self, mode):
        d = {}
        if mode == "single":
            d.update({name: self.mat_frms["single"][name].get("1.0", "end-1c") for name in ['S', 'k', 'rho']})
        else:
            for leg in ['n', 'p']:
                d.update({f'{leg}_{name}': self.mat_frms[leg][name].get("1.0", "end-1c") for name in ['S', 'k', 'rho']})
        d.update({'Tmax': self.tmax_entry.get(), 'Tmax_unit': self.tmax_unit_var.get(),
                  'Tc': self.tc_entry.get(), 'Tc_unit': self.tc_unit_var.get(),
                  'Qin': self.qin_entry.get(), 'use_resist': self.use_resist_var.get()})
        if hasattr(self, 'l_entry'): d.update({'L': self.l_entry.get(), 'L_unit': self.lmax_unit_var.get()})
        if mode == "single":
            if hasattr(self, 'gamma_h_entry'): d.update({'gamma_h': self.gamma_h_entry.get(), 'gamma_h_unit': self.gamma_h_unit_var.get()})
            if hasattr(self, 'gamma_c_entry'): d.update({'gamma_c': self.gamma_c_entry.get(), 'gamma_c_unit': self.gamma_c_unit_var.get()})
        else:
            for key in ['gamma_nh', 'gamma_nc', 'gamma_ph', 'gamma_pc']:
                if hasattr(self, f"{key}_entry"):
                    d.update({key: getattr(self, f"{key}_entry").get(), key + "_unit": getattr(self, f"{key}_unit_var").get()})
        self._input_cache[mode] = d

    def restore_inputs(self, mode):
        d = self._input_cache.get(mode, {})
        if mode == "single":
            for name in ['S', 'k', 'rho']:
                self.mat_frms["single"][name].delete("1.0", "end")
                self.mat_frms["single"][name].insert("1.0", d.get(name, default_rawdata["single"][name]))
        else:
            for leg in ['n', 'p']:
                for name in ['S', 'k', 'rho']:
                    self.mat_frms[leg][name].delete("1.0", "end")
                    self.mat_frms[leg][name].insert("1.0", d.get(f"{leg}_{name}", default_rawdata[leg][name]))
        self.tmax_entry.delete(0, "end"); self.tmax_entry.insert(0, d.get("Tmax", "873" if mode == "single" else "600")); self.tmax_unit_var.set(d.get("Tmax_unit", "K"))
        self.tc_entry.delete(0, "end"); self.tc_entry.insert(0, d.get("Tc", "300")); self.tc_unit_var.set(d.get("Tc_unit", "K"))
        self.qin_entry.delete(0, "end"); self.qin_entry.insert(0, d.get("Qin", "1")); self.use_resist_var.set(d.get("use_resist", "No"))
        self.build_resist_inputs(); self.toggle_resist_area()
        if hasattr(self, 'l_entry'): self.l_entry.delete(0, "end"); self.l_entry.insert(0, d.get("L", "1"))
        if hasattr(self, 'lmax_unit_var'): self.lmax_unit_var.set(d.get("L_unit", "mm"))
        if mode == "single":
            if hasattr(self, 'gamma_h_entry'): self.gamma_h_entry.delete(0, "end"); self.gamma_h_entry.insert(0, d.get("gamma_h", "1E-8")); self.gamma_h_unit_var.set(d.get("gamma_h_unit", "Ω·m²"))
            if hasattr(self, 'gamma_c_entry'): self.gamma_c_entry.delete(0, "end"); self.gamma_c_entry.insert(0, d.get("gamma_c", "1E-8")); self.gamma_c_unit_var.set(d.get("gamma_c_unit", "Ω·m²"))
        else:
            for key in ['gamma_nh', 'gamma_nc', 'gamma_ph', 'gamma_pc']:
                if hasattr(self, f"{key}_entry"):
                    getattr(self, f"{key}_entry").delete(0, "end"); getattr(self, f"{key}_entry").insert(0, d.get(key, "1E-8"))
                if hasattr(self, f"{key}_unit_var"):
                    getattr(self, f"{key}_unit_var").set(d.get(key + "_unit", "Ω·m²"))

    def build_resist_inputs(self):
        for w in self.resist_frame.winfo_children(): w.destroy()
        row = 0
        tk.Label(self.resist_frame, text="Lmax (Max leg length)", font=("Microsoft YaHei", 11), anchor="w", width=22).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        self.l_entry = tk.Entry(self.resist_frame, width=12, font=("Microsoft YaHei", 12)); self.l_entry.grid(row=row, column=1, sticky="w", padx=2)
        self.lmax_unit_var = tk.StringVar(value="mm"); ttk.Combobox(self.resist_frame, textvariable=self.lmax_unit_var, values=["mm", "m"], width=4, state="readonly").grid(row=row, column=2, sticky="w")
        row += 1
        if self.mode_var.get() == "single":
            tk.Label(self.resist_frame, text="γ,h (Hot-side contact resistivity)", font=("Microsoft YaHei", 11)).grid(row=row, column=0, sticky="w", padx=(5,2))
            self.gamma_h_entry = tk.Entry(self.resist_frame, width=12, font=("Microsoft YaHei", 12)); self.gamma_h_entry.grid(row=row, column=1, sticky="w")
            self.gamma_h_unit_var = tk.StringVar(value="Ω·m²"); ttk.Combobox(self.resist_frame, textvariable=self.gamma_h_unit_var, values=["Ω·m²", "Ω·cm²", "Ω·mm²"], width=8, state="readonly").grid(row=row, column=2, sticky="w", padx=(0,40))
            tk.Label(self.resist_frame, text="γ,c (Cold-side contact resistivity)", font=("Microsoft YaHei", 11)).grid(row=row, column=3, sticky="w", padx=(5,2))
            self.gamma_c_entry = tk.Entry(self.resist_frame, width=12, font=("Microsoft YaHei", 12)); self.gamma_c_entry.grid(row=row, column=4, sticky="w")
            self.gamma_c_unit_var = tk.StringVar(value="Ω·m²"); ttk.Combobox(self.resist_frame, textvariable=self.gamma_c_unit_var, values=["Ω·m²", "Ω·cm²", "Ω·mm²"], width=8, state="readonly").grid(row=row, column=5, sticky="w", padx=(0,10))
        else:
            tk.Label(self.resist_frame, text="γ,n,h (N-leg hot-side contact resistivity)", font=("Microsoft YaHei", 11)).grid(row=row, column=0, sticky="w", padx=(5,2)); self.gamma_nh_entry = tk.Entry(self.resist_frame, width=12, font=("Microsoft YaHei", 12)); self.gamma_nh_entry.grid(row=row, column=1, sticky="w"); self.gamma_nh_unit_var = tk.StringVar(value="Ω·m²"); ttk.Combobox(self.resist_frame, textvariable=self.gamma_nh_unit_var, values=["Ω·m²", "Ω·cm²", "Ω·mm²"], width=8, state="readonly").grid(row=row, column=2, sticky="w", padx=(0,40))
            tk.Label(self.resist_frame, text="γ,n,c (N-leg cold-side contact resistivity)", font=("Microsoft YaHei", 11)).grid(row=row, column=3, sticky="w", padx=(5,2)); self.gamma_nc_entry = tk.Entry(self.resist_frame, width=12, font=("Microsoft YaHei", 12)); self.gamma_nc_entry.grid(row=row, column=4, sticky="w"); self.gamma_nc_unit_var = tk.StringVar(value="Ω·m²"); ttk.Combobox(self.resist_frame, textvariable=self.gamma_nc_unit_var, values=["Ω·m²", "Ω·cm²", "Ω·mm²"], width=8, state="readonly").grid(row=row, column=5, sticky="w", padx=(0,40))
            row += 1
            tk.Label(self.resist_frame, text="γ,p,h (P-leg hot-side contact resistivity)", font=("Microsoft YaHei", 11)).grid(row=row, column=0, sticky="w", padx=(5,2)); self.gamma_ph_entry = tk.Entry(self.resist_frame, width=12, font=("Microsoft YaHei", 12)); self.gamma_ph_entry.grid(row=row, column=1, sticky="w"); self.gamma_ph_unit_var = tk.StringVar(value="Ω·m²"); ttk.Combobox(self.resist_frame, textvariable=self.gamma_ph_unit_var, values=["Ω·m²", "Ω·cm²", "Ω·mm²"], width=8, state="readonly").grid(row=row, column=2, sticky="w", padx=(0,40))
            tk.Label(self.resist_frame, text="γ,p,c (P-leg cold-side contact resistivity)", font=("Microsoft YaHei", 11)).grid(row=row, column=3, sticky="w", padx=(5,2)); self.gamma_pc_entry = tk.Entry(self.resist_frame, width=12, font=("Microsoft YaHei", 12)); self.gamma_pc_entry.grid(row=row, column=4, sticky="w"); self.gamma_pc_unit_var = tk.StringVar(value="Ω·m²"); ttk.Combobox(self.resist_frame, textvariable=self.gamma_pc_unit_var, values=["Ω·m²", "Ω·cm²", "Ω·mm²"], width=8, state="readonly").grid(row=row, column=5, sticky="w", padx=(0,10))

    def toggle_resist_area(self):
        if self.use_resist_var.get() != "Yes":
            self.resist_frame.grid_remove()
        else:
            self.resist_frame.grid()

    def switch_mode(self, target_mode):
        self.save_inputs(self._last_mode)
        self.last_results = None
        self._last_mode = target_mode
        for c in self.material_frame.winfo_children(): c.pack_forget()
        if target_mode == "single":
            self.material_frame.winfo_children()[0].pack(fill="x", padx=4, pady=(3,2))
        else:
            self.material_frame.winfo_children()[1].pack(fill="x", padx=4, pady=(3,2))
            self.material_frame.winfo_children()[2].pack(fill="x", padx=4, pady=(3,2))
        self.build_resist_inputs(); self.restore_inputs(target_mode)

    def open_material_lib(self):
        MaterialLibraryDialog(self).wait_window()

    # ---------- Compute (Corrected with Reciprocal Alpha Logic) ----------
    def on_calc(self):
        self.last_results = None
        try:
            mode = self.mode_var.get()
            Tmax = unit_convert_temp(self.tmax_entry.get(), self.tmax_unit_var.get())
            Tc = unit_convert_temp(self.tc_entry.get(), self.tc_unit_var.get())
            Qin = float(self.qin_entry.get())
            use_resist = self.use_resist_var.get() == "Yes"
            Lmax = unit_convert_length(self.l_entry.get(), self.lmax_unit_var.get()) if use_resist else 1.0

            if mode == "single":
                s_poly = parse_input(self.mat_frms["single"]['S'].get("1.0", "end"), "S")
                k_poly = parse_input(self.mat_frms["single"]['k'].get("1.0", "end"), "k")
                r_poly = parse_input(self.mat_frms["single"]['rho'].get("1.0", "end"), "rho")
                gamma_c_h, gamma_c_c = (unit_convert_resist(self.gamma_h_entry.get(), self.gamma_h_unit_var.get()),
                                        unit_convert_resist(self.gamma_c_entry.get(), self.gamma_c_unit_var.get())) if use_resist else (0, 0)
                
                # alpha here is A/H (Conductance factor), as returned by kernel
                eff, m_opt, alpha, Vopt, Th_sc, Isc, R_star_opt, R_L_opt = run_calc_single(s_poly, k_poly, r_poly, Qin, Tc, Tmax, Lmax, gamma_c_h, gamma_c_c)
                
                Th_oc = fsolve(lambda Th: alpha * safe_polyint_k(k_poly, Tc, Th) - Qin if Th > Tc else Qin, Tmax)[0]
                Voc = abs(np.polyval(np.polyint(s_poly), Th_oc) - np.polyval(np.polyint(s_poly), Tc))
                Th_opt, Iopt = self._solve_Th_I_at_m_single(m_opt, Tc, Qin, Lmax, gamma_c_h, gamma_c_c, s_poly, k_poly, r_poly, alpha)
                
                self.last_results = {"mode":"single","s_poly":s_poly,"k_poly":k_poly,"r_poly":r_poly,
                                     "Qin":Qin,"Tc":Tc,"Tmax":Tmax,"Lmax":Lmax,
                                     "gamma_c_h":gamma_c_h,"gamma_c_c":gamma_c_c,
                                     "alpha":alpha,"m_opt":m_opt,"Th_sc":Th_sc,"Th_oc":Th_oc,"Voc":Voc,
                                     "eff":eff,"Vopt":Vopt,"Isc":Isc,"R_star_opt":R_star_opt,"R_L_opt":R_L_opt,
                                     "Th_opt":Th_opt,"I_opt":Iopt}
                eff_percent = eff * 100
                Pmax = Qin * eff
                
                # [DISPLAY FIX]: Output 1/alpha to match paper's H/A definition
                alpha_paper = 1.0 / alpha if alpha != 0 else 0
                
                result_lines = [
                    f"Maximum conversion efficiency:  ηₘₐₓ = {eff_percent:.5g}%     Maximum output power: Pₘₐₓ = {Pmax:.5g} W",
                    f"Optimal load ratio: mₒₚₜ = {m_opt:.5g}     Optimal load voltage: Vₒₚₜ = {Vopt:.5g} V     Optimal load resistance: Rₒₚₜ = {R_L_opt:.5g} Ω",
                    f"Optimal length-to-area ratio (H/A): αₒₚₜ = {alpha_paper:.5g} m⁻¹"
                ]
                if use_resist:
                    # Internal alpha is A/H, so Area = alpha * L
                    result_lines.append(f"Optimal cross-sectional area: Aₒₚₜ = {alpha * Lmax * 1e6:.5g} mm² (L=Lmax)")
                result_lines.extend(["", "[Open/short circuit at optimized size (The Open-circuit results are unreliable cause they exceed the material limits.)]",
                                     f"Open-circuit hot-side temperature: Th,oc = {Th_oc:.5g} K     Open-circuit voltage: Voc = {Voc:.5g} V",
                                     f"Short-circuit hot-side temperature: Th,sc = {Th_sc:.5g} K     Short-circuit current: Isc = {Isc:.5g} A"])
                result_str = "\n".join(result_lines)

            else:  # Couple
                p_s_poly = parse_input(self.mat_frms["p"]['S'].get("1.0", "end"), "S")
                p_k_poly = parse_input(self.mat_frms["p"]['k'].get("1.0", "end"), "k")
                p_r_poly = parse_input(self.mat_frms["p"]['rho'].get("1.0", "end"), "rho")
                n_s_poly = parse_input(self.mat_frms["n"]['S'].get("1.0", "end"), "S")
                n_k_poly = parse_input(self.mat_frms["n"]['k'].get("1.0", "end"), "k")
                n_r_poly = parse_input(self.mat_frms["n"]['rho'].get("1.0", "end"), "rho")
                if use_resist:
                    rc_ph = unit_convert_resist(self.gamma_ph_entry.get(), self.gamma_ph_unit_var.get())
                    rc_pc = unit_convert_resist(self.gamma_pc_entry.get(), self.gamma_pc_unit_var.get())
                    rc_nh = unit_convert_resist(self.gamma_nh_entry.get(), self.gamma_nh_unit_var.get())
                    rc_nc = unit_convert_resist(self.gamma_nc_entry.get(), self.gamma_nc_unit_var.get())
                else:
                    rc_ph = rc_pc = rc_nh = rc_nc = 0
                
                # Internal alphas are A/H
                eff, m_opt, alpha, beta, alphaP, alphaN, Pmax, Vopt, Th_sc, Isc, R_star_opt, R_L_opt = run_calc_Couple(
                    Qin, Tc, Tmax, Lmax, rc_nh, rc_nc, rc_ph, rc_pc,
                    p_s_poly, p_k_poly, p_r_poly, n_s_poly, n_k_poly, n_r_poly
                )

                def eq_oc_Couple(Th):
                    if Th <= Tc: return Qin
                    return (alphaP * safe_polyint_k(p_k_poly, Tc, Th) + alphaN * safe_polyint_k(n_k_poly, Tc, Th)) - Qin

                Th_oc = fsolve(eq_oc_Couple, Tmax)[0]
                Sum_pst, Sum_nst = np.polyint(p_s_poly), np.polyint(n_s_poly)
                Voc = (np.polyval(Sum_pst, Th_oc) - np.polyval(Sum_pst, Tc)) - (np.polyval(Sum_nst, Th_oc) - np.polyval(Sum_nst, Tc))

                Th_opt, Iopt = self._solve_Th_I_at_m_Couple(
                    m_opt, beta, Tc, Qin, Lmax, rc_nh, rc_nc, rc_ph, rc_pc,
                    p_s_poly, p_k_poly, p_r_poly, n_s_poly, n_k_poly, n_r_poly, alphaP
                )

                self.last_results = {
                    "mode": "Couple",
                    "p_s_poly": p_s_poly, "p_k_poly": p_k_poly, "p_r_poly": p_r_poly,
                    "n_s_poly": n_s_poly, "n_k_poly": n_k_poly, "n_r_poly": n_r_poly,
                    "Qin": Qin, "Tc": Tc, "Tmax": Tmax, "Lmax": Lmax,
                    "rc_ph": rc_ph, "rc_pc": rc_pc, "rc_nh": rc_nh, "rc_nc": rc_nc,
                    "alpha": alpha, "alphaP": alphaP, "alphaN": alphaN, "beta": beta,
                    "m_opt": m_opt, "Th_sc": Th_sc, "Th_oc": Th_oc, "Voc": Voc,
                    "eff": eff, "Pmax": Pmax, "Vopt": Vopt, "Isc": Isc, "R_star_opt": R_star_opt, "R_L_opt": R_L_opt,
                    "Th_opt": Th_opt, "I_opt": Iopt
                }

                eff_percent = eff * 100
                
                # [DISPLAY FIX]: Output 1/alpha to match paper's H/A definition
                alpha_paper = 1.0 / alpha if alpha != 0 else 0
                alphaP_paper = 1.0 / alphaP if alphaP != 0 else 0
                alphaN_paper = 1.0 / alphaN if alphaN != 0 else 0
                
                result_lines = [
                    f"Max conversion efficiency:  ηₘₐₓ = {eff_percent:.5g}%     Max output power: Pₘₐₓ = {Pmax:.5g} W",
                    f"Optimal load ratio: mₒₚₜ = {m_opt:.5g}     Optimal load voltage: Vₒₚₜ = {Vopt:.5g} V     Optimal load resistance: Rₒₚₜ = {R_L_opt:.5g} Ω",
                    f"Optimal N/P area ratio: βₒₚₜ = {beta:.5g}",
                    f"Optimal length-to-area ratio (Total): αₒₚₜ = {alpha_paper:.5g} m⁻¹",
                    f"P-leg H/A: αₚ,ₒₚₜ = {alphaP_paper:.5g} m⁻¹      N-leg H/A: αₙ,ₒₚₜ = {alphaN_paper:.5g} m⁻¹"
                ]
                if use_resist:
                    # Internal alphas are A/H
                    areaP = alphaP * Lmax
                    areaN = alphaN * Lmax
                    result_lines.append(f"Optimal total cross-section: Aₒₚₜ = {(areaP+areaN) * 1e6:.5g} mm² (L = Lmax)")
                    result_lines.append(f"  - P-leg Aₚ,ₒₚₜ = {areaP * 1e6:.5g} mm²      - N-leg Aₙ,ₒₚₜ = {areaN * 1e6:.5g} mm²")
                result_lines.extend([
                    "",
                    "[Open/short circuit at optimized size (The Open-circuit results are unreliable cause they exceed the material limits.)]",
                    f"Open-circuit hot-side temp: Th,oc = {Th_oc:.5g} K     Open-circuit voltage: Voc = {Voc:.5g} V",
                    f"Short-circuit hot-side temp: Th,sc = {Th_sc:.5g} K     Short-circuit current: Isc = {Isc:.5g} A"
                ])
                result_str = "\n".join(result_lines)

            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", result_str)
            self.result_text.config(state="disabled")
        except Exception as e:
            self.last_results = None
            messagebox.showerror("Error", f"Computation failed: {e}")

    # — Solve Th and I at m_opt (single leg)
    def _solve_Th_I_at_m_single(self, m, Tc, Qin, Lmax, gamma_c_h, gamma_c_c, s_poly, k_poly, r_poly, alpha):
        Sum_st, Sum_rt = np.polyint(s_poly), np.polyint(r_poly)
        p_taut = np.convolve(np.polyder(s_poly), [1, 0]); Sum_taut = np.polyint(p_taut)
        def heat_balance(Th):
            if Th <= Tc + 1e-6: return -Qin
            dt = Th - Tc
            SengP = np.polyval(Sum_st, Th) - np.polyval(Sum_st, Tc)
            KengP = safe_polyint_k(k_poly, Tc, Th)
            RengP = np.polyval(Sum_rt, Th) - np.polyval(Sum_rt, Tc)
            RengP1 = RengP + dt * (gamma_c_h + gamma_c_c) / Lmax
            int_taut, _ = quad(lambda T: np.polyval(Sum_taut, Th) - np.polyval(Sum_taut, T), Tc, Th, limit=120)
            int_rt, _ = quad(lambda T: np.polyval(Sum_rt, Th) - np.polyval(Sum_rt, T), Tc, Th, limit=120)
            den_m = (KengP
                     + 1/(1+m)*(dt*SengP*Th*np.polyval(s_poly, Th) - SengP*int_taut)/RengP1
                     - 1/(1+m)**2 * SengP**2 * (int_rt +  dt**2 * gamma_c_h / Lmax)/(RengP1**2))
            return alpha * den_m - Qin
        Th_opt = fsolve(heat_balance, Tc + 50)[0]
        dt = Th_opt - Tc
        SengP = np.polyval(Sum_st, Th_opt) - np.polyval(Sum_st, Tc)
        RengP = np.polyval(Sum_rt, Th_opt) - np.polyval(Sum_rt, Tc)
        RengP1 = RengP + dt * (gamma_c_h + gamma_c_c) / Lmax
        R_star = (1 / alpha) * (1 / dt) * RengP1 if (alpha != 0 and dt != 0) else 0
        I = abs(SengP / (R_star * (1 + m))) if R_star != 0 else 0
        return Th_opt, I

    # — Solve Th and I at m_opt (single Couple)
    def _solve_Th_I_at_m_Couple(self, m, beta, Tc, Qin, Lmax, rc_nh, rc_nc, rc_ph, rc_pc,
                               p_s, p_k, p_r, n_s, n_k, n_r, alphaP):
        Sum_pst, Sum_prt = np.polyint(p_s), np.polyint(p_r)
        Sum_nst, Sum_nrt = np.polyint(n_s), np.polyint(n_r)
        p_taut = np.convolve(np.polyder(p_s), [1, 0]); Sum_ptaut = np.polyint(p_taut)
        n_taut = np.convolve(np.polyder(n_s), [1, 0]); Sum_ntaut = np.polyint(n_taut)
        def heat_balance(Th):
            if Th <= Tc + 1e-6: return -Qin
            dt = Th - Tc
            SengP = np.polyval(Sum_pst, Th) - np.polyval(Sum_pst, Tc)
            SengN = np.polyval(Sum_nst, Th) - np.polyval(Sum_nst, Tc)
            KengP = safe_polyint_k(p_k, Tc, Th)
            KengN = safe_polyint_k(n_k, Tc, Th)
            RengP = np.polyval(Sum_prt, Th) - np.polyval(Sum_prt, Tc)
            RengN = np.polyval(Sum_nrt, Th) - np.polyval(Sum_nrt, Tc)
            RengP1 = RengP + dt * (rc_ph + rc_pc) / Lmax
            RengN1 = RengN + dt * (rc_nh + rc_nc) / Lmax
            int_ptaut, _ = quad(lambda T: np.polyval(Sum_ptaut, Th) - np.polyval(Sum_ptaut, T), Tc, Th, limit=120)
            int_ntaut, _ = quad(lambda T: np.polyval(Sum_ntaut, Th) - np.polyval(Sum_ntaut, T), Tc, Th, limit=120)
            int_prt, _ = quad(lambda T: np.polyval(Sum_prt, Th) - np.polyval(Sum_prt, T), Tc, Th, limit=120)
            int_nrt, _ = quad(lambda T: np.polyval(Sum_nrt, Th) - np.polyval(Sum_nrt, T), Tc, Th, limit=120)
            den_common = (1 + m) * (RengP1 + beta**-1 * RengN1)
            den_common_sq = (1 + m)**2 * (RengP1 + beta**-1 * RengN1)**2
            S_diff = SengP - SengN
            denP = KengP + (S_diff * Th * np.polyval(p_s, Th) * dt - S_diff * int_ptaut) / den_common \
                - S_diff**2 * (int_prt + dt**2 * rc_ph / Lmax) / den_common_sq
            denN = KengN - (S_diff * Th * np.polyval(n_s, Th) * dt - S_diff * int_ntaut) / ((1+m)*(beta*RengP1 + RengN1)) \
                - S_diff**2 * (int_nrt + dt**2 * rc_nh / Lmax) / ((1+m)**2 * (beta*RengP1 + RengN1)**2)
            return alphaP * (denP + beta * denN) - Qin
        Th_opt = fsolve(heat_balance, Tc + 50)[0]
        dt = Th_opt - Tc
        SengP = np.polyval(Sum_pst, Th_opt) - np.polyval(Sum_pst, Tc)
        SengN = np.polyval(Sum_nst, Th_opt) - np.polyval(Sum_nst, Tc)
        RengP = np.polyval(Sum_prt, Th_opt) - np.polyval(Sum_prt, Tc)
        RengN = np.polyval(Sum_nrt, Th_opt) - np.polyval(Sum_nrt, Tc)
        RengP1 = RengP + dt * (rc_ph + rc_pc) / Lmax
        RengN1 = RengN + dt * (rc_nh + rc_nc) / Lmax
        alphaN = alphaP * beta
        R_star = (RengP1 / alphaP + RengN1 / alphaN) / dt if (dt!=0 and alphaP!=0 and alphaN!=0) else 0
        Voc_local = SengP - SengN
        I = Voc_local / (R_star * (1 + m)) if R_star != 0 else 0
        return Th_opt, I

    # ---------- Curves ----------
    def on_generate_curves(self):
        if self.last_results is None:
            messagebox.showinfo("Info", "Please run \"Run Optimization\" successfully first.")
            return
        try:
            if self.last_results["mode"] == "single":
                data = self._generate_single_leg_curve_data()
            else:
                data = self._generate_Couple_curve_data()
            if data is None or len(data['v_load']) < 3:
                messagebox.showerror("Error", "Insufficient data points to plot curves.\nPlease check if your inputs are physically reasonable.")
                return
            self._plot_curves(data)
        except Exception as e:
            messagebox.showerror("Error generating curves", f"Unable to generate load curves.\nOriginal error: {e}")

    def _generate_single_leg_curve_data(self):
        p = self.last_results
        s_poly, k_poly, r_poly = p['s_poly'], p['k_poly'], p['r_poly']
        Qin, Tc, Lmax = p['Qin'], p['Tc'], p['Lmax']
        gamma_c_h, gamma_c_c = p['gamma_c_h'], p['gamma_c_c']
        alpha, Th_sc, Th_oc = p['alpha'], p['Th_sc'], p['Th_oc']
        Sum_st, Sum_rt = np.polyint(s_poly), np.polyint(r_poly)
        p_taut = np.convolve(np.polyder(s_poly), [1, 0]); Sum_taut = np.polyint(p_taut)

        def heat_balance_eq(m, Th):
            if m < 0: return Qin
            if Th <= Tc + 1e-6: return -Qin
            dt = Th - Tc
            SengP = np.polyval(Sum_st, Th) - np.polyval(Sum_st, Tc)
            KengP = safe_polyint_k(k_poly, Tc, Th)
            RengP = np.polyval(Sum_rt, Th) - np.polyval(Sum_rt, Tc)
            RengP1 = RengP + dt * (gamma_c_h + gamma_c_c) / Lmax
            if abs(RengP1) < 1e-12: return alpha * KengP - Qin
            int_taut, _ = quad(lambda T: np.polyval(Sum_taut, Th) - np.polyval(Sum_taut, T), Tc, Th, limit=120)
            int_rt, _ = quad(lambda T: np.polyval(Sum_rt, Th) - np.polyval(Sum_rt, T), Tc, Th, limit=120)
            den_m = (KengP
                     + 1/(1+m)*(dt*SengP*Th*np.polyval(s_poly, Th) - SengP*int_taut)/RengP1
                     - 1/(1+m)**2 * SengP**2 * (int_rt + dt**2 * gamma_c_h / Lmax)/(RengP1**2))
            q_in_calc = alpha * den_m
            return q_in_calc - Qin

        Th_range = np.linspace(Th_sc, Th_oc, 50)
        results = {'th': [], 'v_load': [], 'eta': [], 'm': [], 'i': [], 'rstar': []}

        for Th_val in Th_range:
            try:
                sol = root_scalar(heat_balance_eq, args=(Th_val,), method='brentq', bracket=[1e-12, 1e7])
                if not sol.converged: continue
                m_val = float(sol.root)
                dt = Th_val - Tc
                if dt <= 0: continue

                SengP = np.polyval(Sum_st, Th_val) - np.polyval(Sum_st, Tc)
                RengP = np.polyval(Sum_rt, Th_val) - np.polyval(Sum_rt, Tc)
                RengP1 = RengP + dt * (gamma_c_h + gamma_c_c) / Lmax
                R_star = (1 / alpha) * (1 / dt) * RengP1 if (alpha != 0 and dt != 0) else 0
                if R_star <= 0: continue

                I = abs(SengP / (R_star * (1 + m_val)))
                V_load = I * m_val * R_star
                P_out = I * V_load
                eta = P_out / p['Qin'] * 100.0

                results['th'].append(Th_val)
                results['v_load'].append(V_load)
                results['eta'].append(eta)
                results['m'].append(m_val)
                results['i'].append(I)
                results['rstar'].append(R_star)
            except Exception:
                continue

        if len(results['th']) >= 1:
            results['v_load'] = np.array([0.0] + results['v_load'] + [p['Voc']])
            results['eta']    = np.array([0.0] + results['eta']    + [0.0])
            results['th']     = np.array([p['Th_sc']] + results['th'] + [p['Th_oc']])
            results['m']      = np.array([0.0] + results['m'] + [1e9])
            results['i']      = np.array([p['Isc']] + results['i'] + [0.0])
            results['rstar']  = np.array([p['R_star_opt']] + results['rstar'] + [p['R_star_opt']])
        return results

    def _generate_Couple_curve_data(self):
        p = self.last_results
        p_s, p_k, p_r = p['p_s_poly'], p['p_k_poly'], p['p_r_poly']
        n_s, n_k, n_r = p['n_s_poly'], p['n_k_poly'], p['n_r_poly']
        Qin, Tc, Lmax = p['Qin'], p['Tc'], p['Lmax']
        alphaP, beta_opt = p['alphaP'], p['beta']
        rc_ph, rc_pc, rc_nh, rc_nc = p['rc_ph'], p['rc_pc'], p['rc_nh'], p['rc_nc']
        Th_sc, Th_oc = p['Th_sc'], p['Th_oc']

        Sum_pst, Sum_prt = np.polyint(p_s), np.polyint(p_r)
        Sum_nst, Sum_nrt = np.polyint(n_s), np.polyint(n_r)
        p_taut = np.convolve(np.polyder(p_s), [1, 0]); Sum_ptaut = np.polyint(p_taut)
        n_taut = np.convolve(np.polyder(n_s), [1, 0]); Sum_ntaut = np.polyint(n_taut)

        def heat_balance_eq(m, Th):
            if m < 0: return Qin
            if Th <= Tc + 1e-6: return -Qin
            dt = Th - Tc
            SengP = np.polyval(Sum_pst, Th) - np.polyval(Sum_pst, Tc)
            SengN = np.polyval(Sum_nst, Th) - np.polyval(Sum_nst, Tc)
            KengP = safe_polyint_k(p_k, Tc, Th)
            KengN = safe_polyint_k(n_k, Tc, Th)
            RengP = np.polyval(Sum_prt, Th) - np.polyval(Sum_prt, Tc)
            RengN = np.polyval(Sum_nrt, Th) - np.polyval(Sum_nrt, Tc)
            RengP1 = RengP + dt * (rc_ph + rc_pc) / Lmax
            RengN1 = RengN + dt * (rc_nh + rc_nc) / Lmax
            if abs(RengP1 + beta_opt**-1 * RengN1) < 1e-12:
                return alphaP * (KengP + beta_opt * KengN) - Qin
            int_ptaut, _ = quad(lambda T: np.polyval(Sum_ptaut, Th) - np.polyval(Sum_ptaut, T), Tc, Th, limit=90)
            int_ntaut, _ = quad(lambda T: np.polyval(Sum_ntaut, Th) - np.polyval(Sum_ntaut, T), Tc, Th, limit=90)
            int_prt, _ = quad(lambda T: np.polyval(Sum_prt, Th) - np.polyval(Sum_prt, T), Tc, Th, limit=90)
            int_nrt, _ = quad(lambda T: np.polyval(Sum_nrt, Th) - np.polyval(Sum_nrt, T), Tc, Th, limit=90)
            den_common = (1 + m) * (RengP1 + beta_opt**-1 * RengN1)
            den_common_sq = (1 + m)**2 * (RengP1 + beta_opt**-1 * RengN1)**2
            S_diff = SengP - SengN
            denP = KengP + (S_diff * Th * np.polyval(p_s, Th) * dt - S_diff * int_ptaut) / den_common \
                - S_diff**2 * (int_prt + dt**2 * rc_ph / Lmax) / den_common_sq
            denN = KengN - (S_diff * Th * np.polyval(n_s, Th) * dt - S_diff * int_ntaut) / ((1+m)*(beta_opt*RengP1 + RengN1)) \
                - S_diff**2 * (int_nrt + dt**2 * rc_nh / Lmax) / ((1+m)**2 * (beta_opt*RengP1 + RengN1)**2)
            return alphaP * (denP + beta_opt * denN) - Qin

        Th_range = np.linspace(Th_sc, Th_oc, 50)
        results = {'th': [], 'v_load': [], 'eta': [], 'm': [], 'i': [], 'rstar': []}

        for Th_val in Th_range:
            try:
                sol = root_scalar(heat_balance_eq, args=(Th_val,), method='brentq', bracket=[1e-12, 1e7])
                if not sol.converged: continue
                m_val = float(sol.root)
                dt = Th_val - Tc
                if dt <= 0: continue

                SengP = np.polyval(Sum_pst, Th_val) - np.polyval(Sum_pst, Tc)
                SengN = np.polyval(Sum_nst, Th_val) - np.polyval(Sum_nst, Tc)
                RengP = np.polyval(Sum_prt, Th_val) - np.polyval(Sum_prt, Tc)
                RengN = np.polyval(Sum_nrt, Th_val) - np.polyval(Sum_nrt, Tc)
                RengP1 = np.polyval(Sum_prt, Th_val) - np.polyval(Sum_prt, Tc) + dt * (rc_ph + rc_pc) / Lmax
                RengN1 = np.polyval(Sum_nrt, Th_val) - np.polyval(Sum_nrt, Tc) + dt * (rc_nh + rc_nc) / Lmax
                alphaN = alphaP * beta_opt
                R_star = (RengP1 / alphaP + RengN1 / alphaN) / dt if (dt!=0 and alphaP!=0 and alphaN!=0) else 0
                if R_star <= 0: continue

                Voc_local = SengP - SengN
                I = Voc_local / (R_star * (1 + m_val))
                V_load = I * m_val * R_star
                P_out = I * V_load
                eta = P_out / p['Qin'] * 100.0

                results['th'].append(Th_val)
                results['v_load'].append(V_load)
                results['eta'].append(eta)
                results['m'].append(m_val)
                results['i'].append(I)
                results['rstar'].append(R_star)
            except Exception:
                continue

        if len(results['th']) >= 1:
            results['v_load'] = np.array([0.0] + results['v_load'] + [p['Voc']])
            results['eta']    = np.array([0.0] + results['eta']    + [0.0])
            results['th']     = np.array([p['Th_sc']] + results['th'] + [p['Th_oc']])
            results['m']      = np.array([0.0] + results['m'] + [1e9])
            results['i']      = np.array([p['Isc']] + results['i'] + [0.0])
            results['rstar']  = np.array([p['R_star_opt']] + results['rstar'] + [p['R_star_opt']])
        return results

    def _plot_curves(self, data):
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

        plot_window = tk.Toplevel(self)
        plot_window.title("Load Characteristics Curves")
        plot_window.geometry("900x620")

        titlebar = tk.Frame(plot_window); titlebar.pack(fill="x", padx=10, pady=(8,4))
        tk.Label(titlebar, text="Load characteristics under optimized geometry", font=("Microsoft YaHei", 16, "bold")).pack(side="left")
        export_btn = tk.Button(titlebar, text="Export plot data", font=("Microsoft YaHei", 11, "bold"),
                               bg="#87CEFA", activebackground="#A8DBFA",
                               command=lambda: self._export_plot_data(data))
        export_btn.pack(side="right")

        fig, ax1 = plt.subplots(figsize=(8.5, 5.4), tight_layout=True)

        v_load = np.array(data['v_load'])
        eta    = np.array(data['eta'])
        th_arr = np.array(data['th'])

        ax1.set_xlabel("Load voltage V_load (V)", fontsize=12)
        ax1.set_ylabel("Conversion efficiency η (%)", color='tab:red', fontsize=12)
        p_line, = ax1.plot(v_load, eta, 'o-', color='tab:red', label="Conversion efficiency", markersize=3, picker=5)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True)

        # Optimum point in yellow
        opt_eta = self.last_results['eff'] * 100.0
        opt_v   = self.last_results['Vopt']
        ax1.plot(opt_v, opt_eta, marker='*', markersize=11, markerfacecolor='yellow',
                 markeredgecolor='black', label=f'Computed optimum  η_max = {opt_eta:.4f} %')
        ax1.vlines(x=opt_v, ymin=0, ymax=opt_eta, colors='grey', linestyles='dashed', linewidth=1.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Hot-side temperature Th (K)", color='tab:blue', fontsize=12)
        t_line, = ax2.plot(v_load, th_arr, 'o-', color='tab:blue', label="Hot-side temperature", markersize=3, picker=5)
        t_safe_line = ax2.axhline(y=self.last_results['Tmax'], color='green', linestyle='--',
                                  label=f"T_max_safe = {self.last_results['Tmax']:.1f} K")

        ax1.legend([p_line, ax1.lines[1], t_line, t_safe_line],
                   ["Fixed heat-flux efficiency (optimized geometry)",
                    f"Computed optimum  η_max = {opt_eta:.4f} %",
                    "Hot-side temperature (fixed heat flux)",
                    f"T_max_safe = {self.last_results['Tmax']:.1f} K"],
                   loc='best')

        # ========================== Add: V_out–η dashed curve at fixed Th = Tmax ==========================
        try:
            lr = self.last_results
            m_arr = np.array(data['m'], dtype=float)  # reuse sampled m
            Tc = lr['Tc']; Th_fix = lr['Tmax']; dT = Th_fix - Tc
            if dT > 0 and m_arr.size >= 2:
                if lr['mode'] == 'single':
                    s_poly = lr['s_poly']; k_poly = lr['k_poly']; r_poly = lr['r_poly']
                    Sum_s = np.polyint(s_poly); Sum_r = np.polyint(r_poly)
                    Seng = np.polyval(Sum_s, Th_fix) - np.polyval(Sum_s, Tc)
                    Keng = safe_polyint_k(k_poly, Tc, Th_fix)
                    Reng = np.polyval(Sum_r, Th_fix) - np.polyval(Sum_r, Tc)
                    Reng_plus = Reng + dT * (lr['gamma_c_h'] + lr['gamma_c_c']) / lr['Lmax']
                    taut_poly = np.convolve(np.polyder(s_poly), [1, 0]); Sum_taut = np.polyint(taut_poly)
                    I_tau, _ = quad(lambda T: np.polyval(Sum_taut, Th_fix) - np.polyval(Sum_taut, T), Tc, Th_fix, limit=200)
                    I_rho, _ = quad(lambda T: np.polyval(Sum_r,    Th_fix) - np.polyval(Sum_r,    T), Tc, Th_fix, limit=200)
                    ZT_eng = (Seng**2) * dT / (Keng * Reng_plus) if (Keng>0 and Reng_plus>0) else 0.0
                    s_Th = np.polyval(s_poly, Th_fix)
                    gamma_h = lr['gamma_c_h']; Lmax = lr['Lmax']
                    Voc_fixed = abs(Seng)
                    eps = 1e-16
                    denom_common = dT * (1.0 + m_arr) * (Reng_plus + 0.0)
                    bracket = (1.0 + m_arr) / (ZT_eng + eps) \
                              + (Th_fix * s_Th) / (Seng + eps) \
                              - I_tau / (dT + eps)  / (Seng + eps)\
                              - (I_rho + (dT**2) * gamma_h / max(Lmax, eps)) / np.maximum(denom_common, eps)
                    eta_dash = np.maximum(0.0, (m_arr / (1.0 + m_arr)) / np.maximum(bracket, eps)) * 100.0
                    v_dash = Voc_fixed * (m_arr / (1.0 + m_arr))

                else:  # Couple
                    p_s = lr['p_s_poly']; p_k = lr['p_k_poly']; p_r = lr['p_r_poly']
                    n_s = lr['n_s_poly']; n_k = lr['n_k_poly']; n_r = lr['n_r_poly']
                    Sum_ps = np.polyint(p_s); Sum_pr = np.polyint(p_r)
                    Sum_ns = np.polyint(n_s); Sum_nr = np.polyint(n_r)
                    SengP = np.polyval(Sum_ps, Th_fix) - np.polyval(Sum_ps, Tc)
                    SengN = np.polyval(Sum_ns, Th_fix) - np.polyval(Sum_ns, Tc)
                    KengP = safe_polyint_k(p_k, Tc, Th_fix)
                    KengN = safe_polyint_k(n_k, Tc, Th_fix)
                    RengP = np.polyval(Sum_pr, Th_fix) - np.polyval(Sum_pr, Tc)
                    RengN = np.polyval(Sum_nr, Th_fix) - np.polyval(Sum_nr, Tc)
                    RengP_plus = RengP + dT * (lr['rc_ph'] + lr['rc_pc']) / lr['Lmax']
                    RengN_plus = RengN + dT * (lr['rc_nh'] + lr['rc_nc']) / lr['Lmax']
                    p_taut = np.convolve(np.polyder(p_s), [1, 0]); Sum_ptaut = np.polyint(p_taut)
                    n_taut = np.convolve(np.polyder(n_s), [1, 0]); Sum_ntaut = np.polyint(n_taut)
                    I_tau_P, _ = quad(lambda T: np.polyval(Sum_ptaut, Th_fix) - np.polyval(Sum_ptaut, T), Tc, Th_fix, limit=200)
                    I_tau_N, _ = quad(lambda T: np.polyval(Sum_ntaut, Th_fix) - np.polyval(Sum_ntaut, T), Tc, Th_fix, limit=200)
                    I_rho_P, _ = quad(lambda T: np.polyval(Sum_pr,    Th_fix) - np.polyval(Sum_pr,    T), Tc, Th_fix, limit=200)
                    I_rho_N, _ = quad(lambda T: np.polyval(Sum_nr,    Th_fix) - np.polyval(Sum_nr,    T), Tc, Th_fix, limit=200)
                    dS = SengP - SengN
                    denom_Z = (np.sqrt(max(KengP,0)*max(RengP_plus,0)) + np.sqrt(max(KengN,0)*max(RengN_plus,0)))**2
                    ZT_eng = (dS**2) * dT / denom_Z if denom_Z>0 else 0.0
                    beta = lr['beta']
                    sp_Th = np.polyval(p_s, Th_fix)
                    sn_Th = np.polyval(n_s, Th_fix)
                    Lmax = lr['Lmax']
                    gamma_ph = lr['rc_ph']; gamma_nh = lr['rc_nh']
                    Voc_fixed = abs(dS)
                    eps = 1e-16
                    denom_common = dT * (1.0 + m_arr) * (RengP_plus + beta * RengN_plus)
                    bracket = (1.0 + m_arr) / (ZT_eng + eps) \
                              + (Th_fix * (sp_Th - sn_Th)) / (dS + eps) \
                              - (I_tau_P - I_tau_N) / (dT + eps) /  (dS + eps)  \
                              - (I_rho_P + beta * I_rho_N + (dT**2) * (gamma_ph + beta * gamma_nh) / max(Lmax, eps)) / np.maximum(denom_common, eps)
                    eta_dash = np.maximum(0.0, (m_arr / (1.0 + m_arr)) / np.maximum(bracket, eps)) * 100.0
                    v_dash = Voc_fixed * (m_arr / (1.0 + m_arr))

                dash_line, = ax1.plot(v_dash, eta_dash, linestyle='--', linewidth=1.0, label="Efficiency at fixed Th = Tmax")
                self._dash_curve_export = {"m": m_arr.copy(), "v": v_dash.copy(), "eta": eta_dash.copy(), "Th": np.full_like(m_arr, Th_fix, dtype=float)}

                ax1.legend(
                    [p_line, ax1.lines[1], t_line, t_safe_line, dash_line],
                    ["Efficiency under optimized geometry (fixed heat-flux)",
                     f"Optimum point η_max = {opt_eta:.4f} %",
                     "Hot-side temperature (fixed heat-flux)",
                     f"T_max_safe = {self.last_results['Tmax']:.1f} K",
                     "Efficiency at fixed Th = T_max_safe"],
                    loc='best')
        except Exception:
            pass
        # ========================== end addition ==========================

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def _export_plot_data(self, data):
        """Copy to clipboard: Tab-separated (TSV).
        Export side-by-side: m, Vload, Th, η(%), I, Rstar, [Vload@Th=Tmax, η@Th=Tmax]
        The “optimum point” is appended at the end.
        """
        try:
            cols = ["m", "Vload(V)", "Th(K)", "η(%)", "I(A)", "Rstar(Ω)"]

            has_dash = hasattr(self, "_dash_curve_export") and self._dash_curve_export is not None
            if has_dash:
                cols += ["Vload@Th=Tmax(V)", "η@Th=Tmax(%)"]
                # To also output Th@Tmax, insert:
                # cols.insert(3, "Th@Tmax(K)")

            m_arr = np.array(data['m'])
            V_arr = np.array(data['v_load'])
            T_arr = np.array(data['th'])
            E_arr = np.array(data['eta'])
            I_arr = np.array(data['i'])
            R_arr = np.array(data['rstar'])

            if has_dash:
                d = self._dash_curve_export
                m_dash = np.array(d["m"])
                v_dash = np.array(d["v"])
                e_dash = np.array(d["eta"])
                n = min(len(V_arr), len(v_dash))
            else:
                n = len(V_arr)

            lines = ["\t".join(cols)]
            for i in range(n):
                row_vals = [m_arr[i], V_arr[i], T_arr[i], E_arr[i], I_arr[i], R_arr[i]]
                if has_dash:
                    # If you also export Th@Tmax, insert value here
                    # row_vals.insert(2, Th_fix[i])
                    row_vals += [v_dash[i], e_dash[i]]
                lines.append("\t".join(f"{x:.10g}" for x in row_vals))

            # —— Append “optimum” row (keep alignment)
            opt = self.last_results
            opt_row_base = [
                opt['m_opt'],
                opt['Vopt'],
                opt.get('Th_opt', T_arr[np.argmax(E_arr)] if len(E_arr) else np.nan),
                opt['eff']*100.0,
                opt.get('I_opt', np.nan),
                opt['R_star_opt']
            ]
            if has_dash:
                opt_row = opt_row_base + ["", ""]
            else:
                opt_row = opt_row_base

            lines.append("\t".join(["Optimum", "", "", "", "", ""] + (["", ""] if has_dash else [])))
            lines.append("\t".join(
                f"{v:.10g}" if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else ""
                for v in opt_row
            ))

            txt = "\n".join(lines)
            self.clipboard_clear()
            self.clipboard_append(txt)
            messagebox.showinfo("Copied", "Plot data (including the Th=Tmax fixed-temperature curve) has been copied to the clipboard (TSV). You can paste into Excel.")
        except Exception as e:
            messagebox.showerror("Export failed", f"Failed to export plot data: {e}")

# ============================ Material Library Dialog ============================

class MaterialLibraryDialog(tk.Toplevel):
    def __init__(self, owner: TEGFrame):
        super().__init__(owner)
        self.owner = owner
        self.title("Material Library")
        self.geometry("880x520")
        self.resizable(True, True)

        if owner.mode_var.get() == "single":
                tk.Label(self, text="Double-click a material to apply; or select and use the button below.", font=("Microsoft YaHei", 11)).pack(anchor="w", padx=10, pady=(8,2))
        else:
                tk.Label(self, text="Ctrl-select one P-type and one N-type, then click the button below to apply.", font=("Microsoft YaHei", 11)).pack(anchor="w", padx=10, pady=(8,2))

        self.tree = ttk.Treeview(self, columns=("name","type","tmax"), show="headings", selectmode="extended")
        self.tree.heading("name", text="Name")
        self.tree.heading("type", text="Type")
        self.tree.heading("tmax", text="Tmax(K)")
        self.tree.column("name", width=520); self.tree.column("type", width=60, anchor="center"); self.tree.column("tmax", width=90, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=(0,6))

        footer = tk.Frame(self); footer.pack(fill="x", padx=10, pady=8)
        style = ttk.Style(self)
        if "vista" in style.theme_names():
              style.theme_use("vista")  # rounded on Windows
        style.configure("Lib.TButton", font=("Microsoft YaHei", 11, "bold"), padding=(12, 8))

        if owner.mode_var.get() == "single":
            ttk.Button(footer, text="Save current single-leg material to custom library", command=self.save_current_single_to_custom, style="Lib.TButton").pack(side="left", padx=4)
            tk.Button(footer, text="Apply to single leg", command=self.apply_to_single,
                      font=("Microsoft YaHei", 11, "bold"), bg="#1976D2", fg="white",
                      activebackground="#1565C0", activeforeground="white",
                      relief="raised", bd=2, padx=10, pady=6).pack(side="right", padx=4)
        else:
            ttk.Button(footer, text="Save current P-type to custom library", command=lambda: self.save_leg_to_custom('p'), style="Lib.TButton").pack(side="left", padx=4)
            ttk.Button(footer, text="Save current N-type to custom library", command=lambda: self.save_leg_to_custom('n'), style="Lib.TButton").pack(side="left", padx=4)
            tk.Button(footer, text="Apply to single Couple (as N/P)", command=self.apply_to_Couple,
                      font=("Microsoft YaHei", 11, "bold"), bg="#1976D2", fg="white",
                      activebackground="#1565C0", activeforeground="white",
                      relief="raised", bd=2, padx=10, pady=6).pack(side="right", padx=4)

        self.refresh_list()
        self.tree.bind("<Double-1>", self.on_dblclick)

    def _mats_filtered(self):
        mats = BUILTIN_MATERIALS + load_custom_materials()
        def is_example(m):
            return m.get("__example", False) or m.get("name") == "Custom Example Material (P, 600K)"
        return [m for m in mats if not is_example(m)]

    def refresh_list(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        mats = self._mats_filtered()
        for m in mats:
            self.tree.insert("", "end", values=(m.get("name",""), m.get("type","?"), m.get("Tmax","")))

    def _get_selected_material(self, kind=None):
        sel = self.tree.selection()
        if not sel: return None
        idx = self.tree.index(sel[0])
        mats = self._mats_filtered()
        if idx < len(mats):
            m = mats[idx]
            if kind and m.get("type","").upper() != kind.upper():
                return None
            return m
        return None

    def on_dblclick(self, _):
        if self.owner.mode_var.get() == "single":
            self.apply_to_single()

    def apply_to_single(self):
        m = self._get_selected_material()
        if not m:
            messagebox.showinfo("Info", "Please select a material first.", parent=self); return
        owner = self.owner
        owner.mat_frms["single"]["S"].delete("1.0", "end");  owner.mat_frms["single"]["S"].insert("1.0", m["S"])
        owner.mat_frms["single"]["k"].delete("1.0", "end");  owner.mat_frms["single"]["k"].insert("1.0", m["k"])
        owner.mat_frms["single"]["rho"].delete("1.0", "end"); owner.mat_frms["single"]["rho"].insert("1.0", m["rho"])
        owner.tmax_entry.delete(0, "end"); owner.tmax_entry.insert(0, str(m.get("Tmax", ""))); owner.tmax_unit_var.set("K")
        messagebox.showinfo("Done", f"Applied to single leg, and set Tmax to {m.get('Tmax','?')} K.", parent=self)

    def apply_to_Couple(self):
        sels = self.tree.selection()
        if len(sels) < 2:
            messagebox.showinfo("Info", "Please select one N-type and one P-type material (Ctrl to multi-select).", parent=self); return
        mats = self._mats_filtered()
        pick = []
        for it in sels:
            idx = self.tree.index(it)
            if idx < len(mats): pick.append(mats[idx])
        mN = next((x for x in pick if x.get("type","").upper()=="N"), None)
        mP = next((x for x in pick if x.get("type","").upper()=="P"), None)
        if not (mN and mP):
            messagebox.showinfo("Info", "Make sure the selection includes exactly one N-type and one P-type.", parent=self); return

        owner = self.owner
        owner.mat_frms["n"]["S"].delete("1.0", "end");  owner.mat_frms["n"]["S"].insert("1.0", mN["S"])
        owner.mat_frms["n"]["k"].delete("1.0", "end");  owner.mat_frms["n"]["k"].insert("1.0", mN["k"])
        owner.mat_frms["n"]["rho"].delete("1.0", "end"); owner.mat_frms["n"]["rho"].insert("1.0", mN["rho"])
        owner.mat_frms["p"]["S"].delete("1.0", "end");  owner.mat_frms["p"]["S"].insert("1.0", mP["S"])
        owner.mat_frms["p"]["k"].delete("1.0", "end");  owner.mat_frms["p"]["k"].insert("1.0", mP["k"])
        owner.mat_frms["p"]["rho"].delete("1.0", "end"); owner.mat_frms["p"]["rho"].insert("1.0", mP["rho"])
        try:
            tmin = min(float(mN.get("Tmax", 1e9)), float(mP.get("Tmax", 1e9)))
        except Exception:
            tmin = float(owner.tmax_entry.get() or 0)
        owner.tmax_entry.delete(0, "end"); owner.tmax_entry.insert(0, str(tmin)); owner.tmax_unit_var.set("K")
        messagebox.showinfo("Done", f"Applied to single Couple, and set Tmax to min(N,P) = {tmin} K.", parent=self)

    def save_current_single_to_custom(self):
        owner = self.owner
        S_txt  = owner.mat_frms["single"]["S"].get("1.0", "end-1c").strip()
        k_txt  = owner.mat_frms["single"]["k"].get("1.0", "end-1c").strip()
        rho_txt= owner.mat_frms["single"]["rho"].get("1.0", "end-1c").strip()
        if not (S_txt and k_txt and rho_txt):
            messagebox.showwarning("Warning", "S/k/ρ of the current single-leg material cannot be empty.", parent=self)
            return
        dlg = MaterialMetaDialog(self, need_type=True, default_type="P")
        self.wait_window(dlg)
        if not dlg.result: return
        entry = {"name": dlg.result["name"], "type": dlg.result["type"], "Tmax": dlg.result["Tmax"],
                 "S": S_txt, "k": k_txt, "rho": rho_txt}
        save_material_to_custom_lib(entry)
        messagebox.showinfo("Success", f"Saved to custom library: {entry['name']}", parent=self)
        self.refresh_list()

    def save_leg_to_custom(self, leg):
        owner = self.owner
        box = owner.mat_frms['p' if leg=='p' else 'n']
        S_txt  = box["S"].get("1.0", "end-1c").strip()
        k_txt  = box["k"].get("1.0", "end-1c").strip()
        rho_txt= box["rho"].get("1.0", "end-1c").strip()
        if not (S_txt and k_txt and rho_txt):
            messagebox.showwarning("Warning", f"S/k/ρ of the current {'P' if leg=='p' else 'N'}-type material cannot be empty.", parent=self)
            return
        dlg = MaterialMetaDialog(self, need_type=False)
        self.wait_window(dlg)
        if not dlg.result: return
        entry = {"name": dlg.result["name"], "type": ('P' if leg=='p' else 'N'), "Tmax": dlg.result["Tmax"],
                 "S": S_txt, "k": k_txt, "rho": rho_txt}
        save_material_to_custom_lib(entry)
        messagebox.showinfo("Success", f"Saved to custom library: {entry['name']}", parent=self)
        self.refresh_list()
# ============================ entrance ============================

if __name__ == "__main__":
    ensure_custom_lib()
    MainApp().mainloop()