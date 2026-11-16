import { createContext, useContext } from 'react'
import type { ModelInfo } from './api'
import type { Resource } from './hooks'

export const ModelContext = createContext<Resource<ModelInfo> | null>(null)

export function useModel(): Resource<ModelInfo> | null {
  return useContext(ModelContext)
}
