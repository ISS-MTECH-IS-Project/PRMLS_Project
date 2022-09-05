from PIL import Image
import os
import sys

path = os.getcwd() +'\\ArrowanaRaw\\'
dirs = os.listdir(path)
final_size = 128

def ResizeWithAspect(IsPipeToNN = False):
    for item in dirs:
         # if item == '':
         #     continue
         if os.path.isfile(path+item):
             # get the current image using path
             current_image = Image.open(path+item)
             f, e = os.path.splitext(item)

             # Get size for aspect ratio calc
             current_size = current_image.size
             desired_aspect_ratio = float(final_size) / max(current_size)
             new_image_size = tuple([int(x*desired_aspect_ratio) for x in current_size])

             # resize to a temp holder
             resized_image = current_image.resize(new_image_size, Image.Resampling.LANCZOS)

             # save the in-memory obj to disk
             if not IsPipeToNN:
                 new_image = Image.new("RGB", (final_size, final_size))
                 # centering so use / 2
                 new_image.paste(resized_image, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
                 new_image.save(path + 'resized_' + f + '.jpg', 'JPEG', quality=100)
             else :   # else pipe direct to NN to be implemented
                continue

ResizeWithAspect()
