import os, shutil
from functools import partial

copy_img = {"AF" : partial(shutil.copy2, dst="..\\data\\KDEF_Filtered\\afraid\\"),
            "AN" : partial(shutil.copy2, dst="..\\data\\KDEF_Filtered\\angry\\"),
            "DI" : partial(shutil.copy2, dst="..\\data\\KDEF_Filtered\\disgusted\\"),
            "HA" : partial(shutil.copy2, dst="..\\data\\KDEF_Filtered\\happy\\"),
            "NE" : partial(shutil.copy2, dst="..\\data\\KDEF_Filtered\\neutral\\"),
            "SA" : partial(shutil.copy2, dst="..\\data\\KDEF_Filtered\\sad\\"),
            "SU" : partial(shutil.copy2, dst="..\\data\\KDEF_Filtered\\surprised\\"),}

for dirname, _, filenames in os.walk('..\\data\\KDEF_and_AKDEF\\KDEF'):
    if(dirname.split("\\")[-1][0] == "A"):
        for filename in filenames:
            if filename[6] != "F":
                emotion = filename[4:6]
                img_path = os.path.join(dirname, filename)
                try:
                    copy_img[emotion](img_path)
                except KeyError:
                    print(img_path)
            