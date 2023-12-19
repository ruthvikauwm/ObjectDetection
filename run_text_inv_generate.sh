classes=(airplane bicycle boat bus car dog motorcycle person train truck)
ckpt_paths=(
	textual_inversion/logs/airplane2022-12-17T15-32-26_airplane_run_1/checkpoints/embeddings_gs-6099.pt \
	textual_inversion/logs/automobile2022-12-17T16-14-22_automobile_run_1/checkpoints/embeddings_gs-6099.pt \
	textual_inversion/logs/bird2022-12-17T16-48-07_bird_run_1/checkpoints/embeddings_gs-6099.pt \
	textual_inversion/logs/cat2022-12-17T17-21-34_cat_run_1/checkpoints/embeddings_gs-6099.pt \
	textual_inversion/logs/deer2022-12-17T17-54-57_deer_run_1/checkpoints/embeddings_gs-6099.pt \
	textual_inversion/logs/dog2022-12-17T18-27-38_dog_run_1/checkpoints/embeddings_gs-6099.pt\
	textual_inversion/logs/frog2022-12-17T19-00-42_frog_run_1/checkpoints/embeddings_gs-6099.pt \
	textual_inversion/logs/horse2022-12-17T19-33-50_horse_run_1/checkpoints/embeddings_gs-6099.pt \
	textual_inversion/logs/ship2022-12-17T20-06-57_ship_run_1/checkpoints/embeddings_gs-6099.pt \
	textual_inversion/logs/truck2022-12-17T20-39-55_truck_run_1/checkpoints/embeddings_gs-6099.pt \
)

for var in ${!classes[@]}
do
echo ${classes[$var]}
echo ${ckpt_paths[$var]}
python ./txt2img.py --ddim_eta 0.0 \
                          --n_samples 1000 \
                          --n_iter 1 \
                          --scale 10.0 \
                          --ddim_steps 50 \
                          --embedding_path ${ckpt_paths[$var]} \
                          --ckpt_path models/ldm/text2img-large/model.ckpt \
                          --prompt "a photo of * " \
                          --outdir text_inv_generated_images/${classes[$var]}
done

