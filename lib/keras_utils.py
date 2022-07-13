from operator import itemgetter
import tensorflow as tf
from tensorflow.keras import Model

def describe_layer(layer_type, layer_args, vars_alias=None, is_input=False):
    if vars_alias is None:
        vars_alias = {
            'in': 'in',
            'out': 'out'
        }
    if not is_input:
        description = str(vars_alias['out']) + ' = '
    else:
        description = str(vars_alias['in']) + ' = '
    description += layer_type.__name__
    description += "("
    for param in layer_args.keys():
        if layer_args[param] is not None:
            if description[-1] != '(':
                description += ', '
            if isinstance(layer_args[param], str):
                description += param + '=' + "\"" + layer_args[param] + "\""
            else:
                description += param + '=' + str(layer_args[param])
    description += ')'
    if not is_input:
        description += '(' + str(vars_alias['in']) + ')'
    return description

def build_layer(layer_type, layer_args):
    return layer_type(**layer_args)

def build_model(layers_list, input_vars_names, output_vars_names, model_name, middle_vars={}):
    layer_desc = ''
    try:
        for layer in layers_list:
            if layer[0].__name__ == "Input":
                layer_desc = describe_layer(layer[0], layer[1], {'in':layer[2]}, is_input=True)
                middle_vars[layer[2]] = build_layer(layer[0], layer[1])
            else:
                layer_desc = describe_layer(layer[0], layer[1], {'in':layer[2], 'out':layer[3]}, is_input=False)
                this_layer_input_names = [layer[2]] if not isinstance(layer[2], list) else layer[2]
                this_layer_output_names = [layer[3]] if not isinstance(layer[3], list) else layer[3]
                this_layer_inputs = [middle_vars[key] for key in this_layer_input_names]
                if len(this_layer_inputs) == 1:
                    this_layer_inputs = this_layer_inputs[0]
                for var in this_layer_output_names:
                    if var not in middle_vars:
                        middle_vars[var] = None
                this_layer_outputs = [middle_vars[key] for key in this_layer_output_names]
                this_layer_outputs = build_layer(layer[0], layer[1])(this_layer_inputs)
                if isinstance(this_layer_outputs, list):
                    for i in range(len(this_layer_outputs)):
                        middle_vars[this_layer_output_names[i]] = this_layer_outputs[i]
                else:
                    middle_vars[this_layer_output_names[0]] = this_layer_outputs
    except Exception as err:
        raise Exception(str(err) + "\n" + layer_desc)
    outputs = [middle_vars[key] for key in output_vars_names]
    inputs = [middle_vars[key] for key in input_vars_names]
    if len(outputs) == 1:
        outputs = outputs[0]
    model = Model(inputs, outputs, name=model_name)
    return model

def describe_model(layers_list, input_vars_names, output_vars_names, model_name, middle_vars={}):
    model = []
    for layer in layers_list:
        if layer[0].__name__ == "Input" and layer[2] not in input_vars_names:
            input_vars_names.append(layer[2])
    model.append("### Model " + (input_vars_names[0] if len(input_vars_names) == 1 else str(input_vars_names).replace('\'', '')) + " -> "\
                 + (output_vars_names[0] if len(output_vars_names) == 1 else str(output_vars_names).replace('\'', '')) + " ###")
    for layer in layers_list:
        if layer[-1] != 'SKIP_DESC':
            if len(layer) == 4:
                model.append(describe_layer(layer[0], layer[1],
                                            {'in': str(layer[2]), 'out': str(layer[3]).replace('[', '').replace(']', '')}))
            else:
                model.append(describe_layer(layer[0], layer[1], {'in': str(layer[2])}, True))                
    return model