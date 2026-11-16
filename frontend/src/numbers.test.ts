import { describe, expect, it } from 'vitest'
import { extractArtefacts, lookupValue } from './numbers'

describe('message artefacts', () => {
  it('finds a short code and a long number without conflating them', () => {
    const found = extractArtefacts('WINNER!! Reply CLAIM to 80086 or call 09061701461 to collect.')
    expect(found.map(entry => entry.raw)).toEqual(['80086', '09061701461'])
    expect(found[0].kind).toBe('shortcode')
    expect(found[0].shape).toBe('5-digit short code')
    expect(found[1].kind).toBe('phone')
    expect(found[1].shape).toBe('11-digit number')
  })

  it('keeps international format and reports it for the lookup', () => {
    const found = extractArtefacts('Call +44 7700 900123 now')
    expect(found).toHaveLength(1)
    expect(found[0].international).toBe(true)
    expect(found[0].digits).toBe('447700900123')
    expect(lookupValue(found[0])).toBe('+447700900123')
  })

  it('does not read the digits inside a link as a phone number', () => {
    const found = extractArtefacts('Claim at http://bit.ly/12345678901 today')
    expect(found).toHaveLength(1)
    expect(found[0].kind).toBe('link')
  })

  it('ignores quantities and identifiers outside dialable length', () => {
    const found = extractArtefacts('You saved 250 pounds, order 1234567890123456789 shipped')
    expect(found).toHaveLength(0)
  })

  it('does not read a money amount or a year as a short code', () => {
    expect(extractArtefacts('Claim your free 1000 dollar gift card in 2026')).toHaveLength(0)
  })

  it('reports each artefact once however often it appears', () => {
    const found = extractArtefacts('Call 09061701461 now. Repeat: 09061701461.')
    expect(found).toHaveLength(1)
  })

  it('finds an everyday message to contain nothing contactable', () => {
    expect(extractArtefacts('Running about 10 minutes late, order me a coffee')).toHaveLength(0)
  })
})
