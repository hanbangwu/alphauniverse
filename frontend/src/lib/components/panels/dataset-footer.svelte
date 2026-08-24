<script lang="ts">
  import { getApp } from '$lib/state/app.svelte'

  const app = getApp()
</script>

<footer class="flex shrink-0 flex-col gap-0.5 p-4 font-mono text-xs text-muted-foreground">
  <span>{app.meta.galaxies} galaxies</span>
  <a href={`https://huggingface.co/datasets/${app.dataset}`}>{app.dataset}</a>
</footer>
