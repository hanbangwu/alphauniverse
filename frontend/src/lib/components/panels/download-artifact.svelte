<script lang="ts">
  import { artifactUrl } from '$lib/client'
  import { Button } from '$lib/components/ui/button'
  import { getMeta } from '$lib/state/app.svelte'

  const meta = getMeta()
</script>

<Button href={artifactUrl(meta.embeddings)} variant="ghost" size="sm">Download embeddings</Button>
