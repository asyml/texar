 mpirun -np 2 \
    -H 54.202.134.113:1,54.244.60.228:1\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ens3 \
    python bert_classifier_main.py --do_train --do_eval --do_test --output_dir='multigpu_output'
