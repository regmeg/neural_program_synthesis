Traceback (most recent call last):
  File "/home/rb7e15/2.7v/model.py", line 88, in <module>
    main()
  File "/home/rb7e15/2.7v/model.py", line 72, in main
    run_session_2RNNS(model, cfg, x_train, x_test, y_train, y_test)
  File "/home/rb7e15/2.7v/session.py", line 274, in run_session_2RNNS
    m.use_both_losses: use_both_losses
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 789, in run
    run_metadata_ptr)
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 997, in _run
    feed_dict_string, options, run_metadata)
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1132, in _do_run
    target_list, options, run_metadata)
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1152, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Infinity in summary histogram for: Summaries_norms/RNN_op/Grads/global_norm/global_norm_0/histogram
	 [[Node: Summaries_norms/RNN_op/Grads/global_norm/global_norm_0/histogram = HistogramSummary[T=DT_DOUBLE, _device="/job:localhost/replica:0/task:0/cpu:0"](Summaries_norms/RNN_op/Grads/global_norm/global_norm_0/histogram/tag, RNN_op/Grads/global_norm/global_norm)]]

Caused by op u'Summaries_norms/RNN_op/Grads/global_norm/global_norm_0/histogram', defined at:
  File "/home/rb7e15/2.7v/model.py", line 88, in <module>
    main()
  File "/home/rb7e15/2.7v/model.py", line 71, in main
    model = eval(cfg[u'model']+u"(cfg, ops, mem)")
  File "<string>", line 1, in <module>
  File "/home/rb7e15/2.7v/rnn_base.py", line 68, in __init__
    self.variable_summaries(self.norms)
  File "/home/rb7e15/2.7v/nn_base.py", line 254, in variable_summaries
    tf.summary.histogram(u'histogram', var)
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/summary/summary.py", line 221, in histogram
    tag=scope.rstrip('/'), values=values, name=scope)
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/ops/gen_logging_ops.py", line 131, in _histogram_summary
    name=name)
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 2506, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/home/rb7e15/2.7v/TFenv/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 1269, in __init__
    self._traceback = _extract_stack()

InvalidArgumentError (see above for traceback): Infinity in summary histogram for: Summaries_norms/RNN_op/Grads/global_norm/global_norm_0/histogram
	 [[Node: Summaries_norms/RNN_op/Grads/global_norm/global_norm_0/histogram = HistogramSummary[T=DT_DOUBLE, _device="/job:localhost/replica:0/task:0/cpu:0"](Summaries_norms/RNN_op/Grads/global_norm/global_norm_0/histogram/tag, RNN_op/Grads/global_norm/global_norm)]]

