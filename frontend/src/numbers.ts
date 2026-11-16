
export type Artefact = {
  id: string
  raw: string
  digits: string
  international: boolean
  kind: 'shortcode' | 'phone' | 'link'
  shape: string
}

const URL_PATTERN = /\b((?:https?:\/\/|www\.)[^\s<>"']{3,}|[a-z0-9-]+\.(?:com|net|org|co|uk|info|biz|ly|xyz|top|link|click|shop)(?:\/[^\s<>"']*)?)/gi

const NUMBER_PATTERN = /\+?\d(?:[\d\s()\-.]{2,18}\d)?/g

function describe(digits: string, international: boolean): { kind: Artefact['kind']; shape: string } {
  const length = digits.length
  if (length <= 6) return { kind: 'shortcode', shape: `${length}-digit short code` }
  if (international) return { kind: 'phone', shape: `${length}-digit number in international format` }
  return { kind: 'phone', shape: `${length}-digit number` }
}

export function extractArtefacts(text: string): Artefact[] {
  const found: Artefact[] = []
  const seen = new Set<string>()

  const links = text.match(URL_PATTERN) ?? []
  for (const link of links) {
    const key = `link:${link.toLowerCase()}`
    if (seen.has(key)) continue
    seen.add(key)
    found.push({ id: key, raw: link, digits: '', international: false, kind: 'link', shape: 'Web address' })
  }

  const withoutLinks = text.replace(URL_PATTERN, ' ')
  const candidates = withoutLinks.match(NUMBER_PATTERN) ?? []
  for (const candidate of candidates) {
    const trimmed = candidate.trim()
    const international = trimmed.startsWith('+')
    const digits = trimmed.replace(/\D/g, '')
    if (digits.length < 5 || digits.length > 15) continue
    const key = `num:${international ? '+' : ''}${digits}`
    if (seen.has(key)) continue
    seen.add(key)
    found.push({ id: key, raw: trimmed, digits, international, ...describe(digits, international) })
  }

  return found
}

export function lookupValue(artefact: Artefact): string {
  return artefact.international ? `+${artefact.digits}` : artefact.digits
}
