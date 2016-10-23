import glob, os
from captcha_recognizer import OCR_recognizer

recognizer = OCR_recognizer('models/sintegra_sc_model.ckpt')

def calculate_accuracy(dir, pattern):
    corrects = 0
    total = 0
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        label, ext = os.path.splitext(os.path.basename(pathAndFilename))
        prediction = recognizer.recognize(pathAndFilename)
        if prediction == label:
            corrects += 1
        total += 1
        print "Prediction: " + prediction
        print "Label: " + label
    accuracy = (float(corrects)/float(total)) * 100
    print "The total accuracy was: " + "{:.2f}%".format(accuracy)
    return accuracy

calculate_accuracy(r'./images/test', r'*.png')

