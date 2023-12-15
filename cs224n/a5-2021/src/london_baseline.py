# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
# evaluate on the eval-corpus

def main() -> None:
    """

    """
    from os.path import abspath, dirname
    from src.utils import evaluate_places

    prediction, y_true = list(), list()
    filename = abspath(f'{dirname(abspath(__file__))}/../birth_dev.tsv')
    with open(filename, 'r') as inputs:
        for index, line in enumerate(inputs.readlines()):
            # read line
            line = line.strip()
            x, y = line.split('\t')
            # add to the list
            prediction.append('London')
            y_true.append(y)
    # report
    total, correct = evaluate_places(y_true=y_true, prediction=prediction)
    print(f'correct {correct} out of {total}, {correct * 100 / total:.2f}% correct')


if __name__ == '__main__':
    from sys import path
    from os.path import abspath, dirname

    path.append(abspath(f'{dirname(abspath(__file__))}/..'))
    main()
