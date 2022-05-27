import imageio 

def create_gif(image_list, gif_name, dura):
    # Save them as frames into a gif  
    all_images = []    
    for image_name in image_list:        
        all_images.append(imageio.imread(image_name))      
        imageio.mimsave(gif_name, all_images, 'GIF', duration = dura)     
    return 

image_list=[]
for ni in range(100):
    image_list.append(str(ni)+'.png')

gif_name = 'sin_cos_movie.gif'    

create_gif(image_list, gif_name,0.1)


