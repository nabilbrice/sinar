from PIL import Image
import numpy as np
from jax import Array

def convert_to_pil(frame: Array) -> Image:
    """Converts the frame of colors to a PIL Image.
    
    Parameters
    ----------
    frame : Array
        The RGBA for each pixel, should have shape (xres, yres, 4).
    
    Returns
    -------
    image : Image
        The PIL Image.
    """
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise AssertionError(
            f"Frame must have shape (xres, yres, 3), "
            f"but frame has shape {frame.shape}.\n"
            f"Consider reshaping the frame to match the required format.")

    image = np.abs(np.array(frame))
    image = (image * 250).astype(np.uint8)
    return Image.fromarray(image, mode='RGB')

def save_frame_as_png(frame: Array, filepath = "image.png"):
    """Saves the frame of colors to a png file.
    
    The frame is first converted to a PIL Image.

    Parameters
    ----------
    frame : Array
        The RGBA for each pixel, should have shape (xres, yres, 4).
    filepath : str | optional
        The filepath to save the png formatted image.

    """
    image = convert_to_pil(frame)
    image.save(filepath)

def save_frame_as_gif(frames: list, filepath: str ="animation.gif",
                      duration: int=100, loop: int=0):
    """
    Saves a GIF animation from a sequence of frames.
    
    Parameters
    ----------
    frames : list of arrays
        List of arrays, each representing a frame in the animation
    filepath : str, optional
        Path where the GIF will be saved (default: "animation.gif")
    duration : int, optional
        Duration of each frame in milliseconds (default: 100)
    loop : int, optional
        Number of times to loop the animation. 0 means infinite loop (default: 0)
    """
    # Process each frame into a PIL Image
    pil_frames = [convert_to_pil(frame) for frame in frames]
    
    # Save the sequence as an animated GIF
    if pil_frames:
        pil_frames[0].save(
            filepath,
            format='GIF',
            append_images=pil_frames[1:],
            save_all=True,
            duration=duration,
            loop=loop
        )