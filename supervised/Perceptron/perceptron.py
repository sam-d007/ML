import pandas as pd

#input parameters
w1 = 0.9
w2 = 0.9
bias = -1

#generating and checking the output
test_inputs = [(0,0), (0,1), (1,0), (1,1)]
correct_outputs = [0, 0, 0, 1]
outputs = []

for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = w1*test_input[0] + w2*test_input[1] + bias
    output = linear_combination>=0
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

#Print output
print(outputs)
print("*****")
num_wrong = len([output[4] for output in outputs if output[4]=='No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1','Input 2', 'Linear combination','Activation output', 'Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))

