from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
#import coref_model_MM as cm
#import coref_model_SFA as cm
#import util
#import coref_model_BASE as cm
#import util_BASE as util
#import coref_model_KDGCN_vote2 as cm
#import coref_model_GAT_multigraph as cm
#import coref_model_GAT_new_CRAFT as cm
#import coref_model_GAT_new as cm
#import coref_model_GAT_multi3 as cm
#import coref_model_GAT_vote as cm
#import coref_model_GAT_new_CRAFT as cm
#import coref_model_GAT_multigraph_CRAFT as cm
import coref_model_GAT_nog_CRAFT as cm
#import coref_model_GAT_nograph as cm
#import util_KD_GCN_debug as util
import util_KD_GCN_CRAFT as util

if __name__ == "__main__":

    config = util.initialize_from_env0()

    # Input file in .jsonlines format.
    input_filename = sys.argv[2]

    # Predictions will be written to this file in .jsonlines format.
    output_filename = sys.argv[3]

    model = cm.CorefModel(config)

    with tf.Session() as session:
        model.restore(session)
        with open(output_filename, "w") as output_file:
            with open(input_filename) as input_file:
                for example_num, line in enumerate(input_file.readlines()):
                    example = json.loads(line)
                    tensorized_example = model.tensorize_example(example, is_training=False)
                    feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
                    _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run( model.predictions, feed_dict=feed_dict)
                                       
                    predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
                    
                    example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
                    print(example_num)
                    output_file.write(json.dumps(example))
                    output_file.write("\n")
                    if example_num % 100 == 0:
                        print("Decoded {} examples.".format(example_num + 1))
