<script lang="ts">
  import { artifactUrl } from '$lib/client'
  import { Button } from '$lib/components/ui/button'
</script>

<div class="flex flex-col gap-2">
  <div class="text-xs tracking-wider text-muted-foreground uppercase">Downloads</div>
  <Button href={artifactUrl('encoded')} variant="ghost" size="sm"
    >Download encoded embeddings</Button
  >
  <Button href={artifactUrl('codebook')} variant="ghost" size="sm"
    >Download codebook embeddings</Button
  >
  <Button href={artifactUrl('tokens')} variant="ghost" size="sm">Download tokens</Button>
</div>
