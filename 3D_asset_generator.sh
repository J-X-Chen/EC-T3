for i in {4..9}
 do
    python 3D_glb_generation.py --text "plate0$i" --test-dir .\\glb_assets\\plate0$i --generator_name "Qwen/Qwen-Image" --image_size "1024x1024" --style $i
    python 3D_object_generation.py --name plate0$i --size large --pos 0 0 0.5 --quat 0.70710678 0.70710678 0 0 --max-convex-hull 16 --overwrite
    #python 3D_glb_generation.py --text "mug0$i" --test-dir .\\glb_assets\\mug0$i --generator_name "Qwen/Qwen-Image" --image_size "1024x1024" --style $i
    #python 3D_object_generation.py --name mug0$i --size middle --pos 0 0 0.5 --quat 0.70710678 0.70710678 0 0 --max-convex-hull 24 --overwrite
    python 3D_glb_generation.py --text "bowl0$i" --test-dir .\\glb_assets\\bowl0$i --generator_name "Qwen/Qwen-Image" --image_size "1024x1024" --style $i
    python 3D_object_generation.py --name bowl0$i --size middle --pos 0 0 0.5 --quat 0.70710678 0.70710678 0 0 --max-convex-hull 24 --overwrite
done
