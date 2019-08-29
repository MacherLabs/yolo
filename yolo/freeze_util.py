import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import os.path
from os import path

#Converts darknet model to Tensor RT graph
def freeze_graph(tfnet,mode='trt',build = False):
    # Check if we have frozen graph already
    
    if path.exists('trt_frozen_yolo.pb') == True and build == False:
        with tf.gfile.FastGFile('trt_frozen_yolo.pb','rb') as f:
            trt_graph_def = tf.GraphDef()
            trt_graph_def.ParseFromString(f.read())
            print('trt graph exists..loaded trt graph..')
    else:
        #Output name for frozen graph
        output_node_names=["output"]
        input_graph_def = tfnet.sess.graph.as_graph_def();
        
        #Do some modeifications in the graph
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        print('freezing graph..')
        
        #Freeze the graph
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                tfnet.sess, # The session
                input_graph_def, # input_graph_def is useful for retrieving the nodes 
                output_node_names)
    
        #Frozen model path
        output_graph="frozen_yolo.pb"
        
        #Save the graph
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print('froze graph..')
            
        print('creating trt graph..')
        #Convert to tensor RT graph
        trt_graph_def=trt.create_inference_graph(input_graph_def= output_graph_def,
                                                max_batch_size=1,
                                                max_workspace_size_bytes=1<<25,
                                                precision_mode='FP16',
                                                minimum_segment_size=5,
                                                maximum_cached_engines=5,
                                                outputs=['output'])
        nodes = [node.name for node in trt_graph_def.node if 'TRTEngineOp' == node.op]
        print('{} Tensort optimization nodes created.'.format(len(nodes)) )
        print('converted to trt graph..')
        
        #Save tensorrt graph
        with tf.gfile.FastGFile("trt_"+output_graph, 'wb') as f:
                f.write(trt_graph_def.SerializeToString())
        print('saved TRT graph')
    
    #If mode is trt return the tensorrt graph
    if mode=='trt':
        output_graph_def= trt_graph_def
      
    # Import the tensort graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(output_graph_def,input_map=None,return_elements=None,name="")
    
    #Create new session to replace the original session
    sess=tf.Session(graph=graph)
    tfnet.sess=sess
    
    #Replace the placeholders of the original graph with tensorRT graph
    tfnet.inp = tfnet.sess.graph.get_tensor_by_name('input:0')
    tfnet.out = tfnet.sess.graph.get_tensor_by_name('output:0')
    
    return tfnet
