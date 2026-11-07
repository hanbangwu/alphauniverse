<script lang="ts">
  import { goto } from '$app/navigation'
  import { resolve } from '$app/paths'
  import markDark from '$lib/assets/alphauniverse-mark-dark.svg'
  import markLight from '$lib/assets/alphauniverse-mark-light.svg'
  import { signIn } from '$lib/client'
  import { Button } from '$lib/components/ui/button'
  import { Input } from '$lib/components/ui/input'

  let password = $state('')
  let pending = $state(false)
  let problem = $state('')

  async function submit(event: SubmitEvent) {
    event.preventDefault()
    pending = true
    const response = await signIn(password).catch(() => undefined)
    pending = false
    if (response?.ok) return goto(resolve('/'))
    problem =
      response?.status === 401
        ? 'Incorrect password.'
        : 'The server is unavailable. Try again in a moment.'
  }
</script>

<main class="grid h-dvh place-content-center gap-6 p-4">
  <h1 class="flex items-center gap-3 text-2xl tracking-wide">
    <img src={markLight} alt="" class="size-12 dark:hidden" />
    <img src={markDark} alt="" class="hidden size-12 dark:block" />
    alphaUniverse
  </h1>
  <form class="grid gap-3" onsubmit={submit}>
    <label class="grid gap-1 text-sm">
      Password
      <Input
        type="password"
        autocomplete="current-password"
        required
        aria-invalid={problem !== ''}
        bind:value={password}
      />
    </label>
    {#if problem}
      <p role="alert" class="text-sm text-destructive">{problem}</p>
    {/if}
    <Button type="submit" disabled={pending}>Enter</Button>
  </form>
</main>
