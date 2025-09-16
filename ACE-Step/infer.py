import click
import os

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler


def sample_data(json_data):
    return (
        json_data["audio_duration"],
        json_data["prompt"],
        json_data["lyrics"],
        json_data["infer_step"],
        json_data["guidance_scale"],
        json_data["scheduler_type"],
        json_data["cfg_type"],
        json_data["omega_scale"],
        ", ".join(map(str, json_data["actual_seeds"])),
        json_data["guidance_interval"],
        json_data["guidance_interval_decay"],
        json_data["min_guidance_scale"],
        json_data["use_erg_tag"],
        json_data["use_erg_lyric"],
        json_data["use_erg_diffusion"],
        ", ".join(map(str, json_data["oss_steps"])),
        json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
        (
            json_data["guidance_scale_lyric"]
            if "guidance_scale_lyric" in json_data
            else 0.0
        ),
    )


@click.command()
@click.option(
    "--checkpoint_path", type=str, default="", help="Path to the checkpoint directory"
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bfloat16")
@click.option(
    "--torch_compile", type=bool, default=False, help="Whether to use torch compile"
)
@click.option(
    "--cpu_offload", type=bool, default=False, help="Whether to use CPU offloading (only load current stage's model to GPU)"
)
@click.option(
    "--overlapped_decode", type=bool, default=False, help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)"
)
@click.option("--device_id", type=int, default=0, help="Device ID to use")
@click.option("--output_path", type=str, default=None, help="Path to save the output")
# Custom options for serverless/batch usage - ALL ACE-Step parameters supported
@click.option("--prompt", type=str, required=True, help="Text prompt for music generation")
@click.option("--duration", type=float, required=True, help="Duration of generated audio in seconds (audio_duration)")
@click.option("--lyrics", type=str, default="", help="Lyrics for the generated music")
@click.option("--infer_step", type=int, default=60, help="Number of inference steps")
@click.option("--guidance_scale", type=float, default=15.0, help="Guidance scale for generation")
@click.option("--scheduler_type", type=str, default="euler", help="Scheduler type (euler, ddim, etc.)")
@click.option("--cfg_type", type=str, default="apg", help="CFG type (apg, single, etc.)")
@click.option("--omega_scale", type=float, default=10.0, help="Omega scale")
@click.option("--manual_seeds", type=str, default="", help="Manual seeds (comma-separated integers)")
@click.option("--guidance_interval", type=float, default=0.5, help="Guidance interval")
@click.option("--guidance_interval_decay", type=float, default=0.0, help="Guidance interval decay")
@click.option("--min_guidance_scale", type=float, default=3.0, help="Minimum guidance scale")
@click.option("--use_erg_tag", is_flag=True, default=False, help="Use ERG tag")
@click.option("--use_erg_lyric", is_flag=True, default=False, help="Use ERG lyric")
@click.option("--use_erg_diffusion", is_flag=True, default=False, help="Use ERG diffusion")
@click.option("--oss_steps", type=str, default="", help="OSS steps (comma-separated integers)")
@click.option("--guidance_scale_text", type=float, default=0.0, help="Guidance scale for text")
@click.option("--guidance_scale_lyric", type=float, default=0.0, help="Guidance scale for lyrics")
def main(checkpoint_path, bf16, torch_compile, cpu_offload, overlapped_decode, device_id, output_path,
         prompt, duration, lyrics, infer_step, guidance_scale, scheduler_type, cfg_type, omega_scale,
         manual_seeds, guidance_interval, guidance_interval_decay, min_guidance_scale, 
         use_erg_tag, use_erg_lyric, use_erg_diffusion, oss_steps, guidance_scale_text, guidance_scale_lyric):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode
    )
    print(f"ACE-Step Pipeline initialized: {model_demo}")

    # Use CLI arguments directly instead of data sampler
    print(f"Generating music with prompt: '{prompt}', duration: {duration}s")
    
    # Convert string parameters to proper formats
    manual_seeds_list = [int(x.strip()) for x in manual_seeds.split(',') if x.strip()] if manual_seeds else [0]
    oss_steps_list = [int(x.strip()) for x in oss_steps.split(',') if x.strip()] if oss_steps else [40]
    
    print(f"Parameters: infer_step={infer_step}, guidance_scale={guidance_scale}, scheduler={scheduler_type}")
    print(f"CFG: {cfg_type}, omega_scale={omega_scale}, manual_seeds={manual_seeds_list}")
    print(f"ERG flags: tag={use_erg_tag}, lyric={use_erg_lyric}, diffusion={use_erg_diffusion}")

    model_demo(
        audio_duration=duration,
        prompt=prompt,
        lyrics=lyrics,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        cfg_type=cfg_type,
        omega_scale=omega_scale,
        manual_seeds=manual_seeds_list,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
        min_guidance_scale=min_guidance_scale,
        use_erg_tag=use_erg_tag,
        use_erg_lyric=use_erg_lyric,
        use_erg_diffusion=use_erg_diffusion,
        oss_steps=oss_steps_list,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        save_path=output_path,
    )
    
    print(f"Audio generation completed successfully! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
