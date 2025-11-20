import bpy
import sys
import os

# --- CONFIGURAÇÕES ---
argv = sys.argv
try:
    index = argv.index("--") + 1
    args = argv[index:]
    INPUT_FILE = args[0]
    OUTPUT_FILE = args[1]
    DECIMATE_RATIO = float(args[2]) if len(args) > 2 else 0.5
except (ValueError, IndexError):
    if bpy.app.background:
        INPUT_FILE = "input.glb"
        OUTPUT_FILE = "output.glb"
        DECIMATE_RATIO = 0.5
    else:
        INPUT_FILE = "input.glb"
        OUTPUT_FILE = "output.glb"
        DECIMATE_RATIO = 0.5

MERGE_DISTANCE = 0.0001 

def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in [bpy.data.meshes, bpy.data.materials, bpy.data.images]:
        for block in col:
            if block.users == 0: col.remove(block)

def process_pipeline():
    reset_scene()
    
    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo não encontrado {INPUT_FILE}")
        sys.exit(1)

    print(f"Importando: {INPUT_FILE}")
    bpy.ops.import_scene.gltf(filepath=INPUT_FILE)
        
    meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not meshes:
        print("ERRO: Nenhuma mesh encontrada.")
        sys.exit(1)

    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active

    print(f"Aplicando Merge by Distance ({MERGE_DISTANCE}m)...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    initial_verts = len(obj.data.vertices)
    bpy.ops.mesh.remove_doubles(threshold=MERGE_DISTANCE)
        
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    final_verts = len(obj.data.vertices)
    print(f"Vértices removidos: {initial_verts - final_verts}")

    # --- PASSO 2: REMOVER ILHAS (NOISE DO TRELLIS) ---
    print("Removendo ilhas desconectadas...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    parts = bpy.context.selected_objects
    if len(parts) > 1:
        # Mantém apenas o objeto com maior volume (Bounding Box)
        largest = max(parts, key=lambda o: o.dimensions.x * o.dimensions.y * o.dimensions.z)
        
        bpy.ops.object.select_all(action='DESELECT')
        count_removed = 0
        for p in parts:
            if p != largest:
                p.select_set(True)
                count_removed += 1
        
        if count_removed > 0:
            print(f"Deletando {count_removed} partes soltas (ruído).")
            bpy.ops.object.delete()
        
        largest.select_set(True)
        bpy.context.view_layer.objects.active = largest
        obj = largest
    else:
        # Se só tinha uma parte, re-seleciona
        parts[0].select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        obj = parts[0]

    # --- PASSO 3: DECIMATE (REDUÇÃO DE PESO) ---
    # Reduz a contagem de faces mantendo UVs
    print(f"Aplicando Decimate (Ratio: {DECIMATE_RATIO})...")
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.ratio = DECIMATE_RATIO
    mod.use_collapse_triangulate = True 
    bpy.ops.object.modifier_apply(modifier="Decimate")
    
    print(f"Contagem final de vértices: {len(obj.data.vertices)}")

    # --- EXPORTAÇÃO ---
    print(f"Exportando para: {OUTPUT_FILE}")
    bpy.ops.export_scene.gltf(
        filepath=OUTPUT_FILE, 
        export_format='GLB',
        use_selection=True,
        # Isso garante que as texturas originais sejam mantidas dentro do GLB
        export_image_format='AUTO'
    )
    
    print("BLENDER_SUCCESS")

if __name__ == "__main__":
    process_pipeline()