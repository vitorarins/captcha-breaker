import glob, os

def rename(dir, pattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        newTitle = title.split(':')[3]
        newName = os.path.join(dir, newTitle + ext)
        print 'Old: ' + pathAndFilename
        print 'New: ' + newName
        os.rename(pathAndFilename, 
                  newName)

rename(r'./images/torename', r'*.png')
