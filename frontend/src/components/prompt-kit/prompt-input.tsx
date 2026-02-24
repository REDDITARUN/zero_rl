"use client"

import { cn } from "@/lib/utils"
import React, { createContext, useContext, useLayoutEffect, useRef, useState } from "react"

type PromptInputContextType = {
  isLoading: boolean
  value: string
  setValue: (value: string) => void
  maxHeight: number | string
  onSubmit?: () => void
  disabled?: boolean
  textareaRef: React.RefObject<HTMLTextAreaElement | null>
}

const PromptInputContext = createContext<PromptInputContextType>({
  isLoading: false,
  value: "",
  setValue: () => {},
  maxHeight: 240,
  onSubmit: undefined,
  disabled: false,
  textareaRef: React.createRef<HTMLTextAreaElement>(),
})

function usePromptInput() {
  return useContext(PromptInputContext)
}

export type PromptInputProps = {
  isLoading?: boolean
  value?: string
  onValueChange?: (value: string) => void
  maxHeight?: number | string
  onSubmit?: () => void
  children: React.ReactNode
  className?: string
  disabled?: boolean
} & React.ComponentProps<"div">

function PromptInput({
  className,
  isLoading = false,
  maxHeight = 240,
  value,
  onValueChange,
  onSubmit,
  children,
  disabled = false,
  onClick,
  ...props
}: PromptInputProps) {
  const [internalValue, setInternalValue] = useState(value || "")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleChange = (newValue: string) => {
    setInternalValue(newValue)
    onValueChange?.(newValue)
  }

  const handleClick: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (!disabled) textareaRef.current?.focus()
    onClick?.(e)
  }

  return (
    <PromptInputContext.Provider
      value={{
        isLoading,
        value: value !== undefined ? value : internalValue,
        setValue: handleChange,
        maxHeight,
        onSubmit,
        disabled,
        textareaRef,
      }}
    >
      <div
        className={cn(
          "relative flex w-full flex-col rounded-2xl border border-white/10 bg-white/5 px-3 py-2 backdrop-blur-sm transition-colors focus-within:border-cyan-500/40 focus-within:ring-1 focus-within:ring-cyan-500/20",
          disabled && "cursor-not-allowed opacity-50",
          className
        )}
        onClick={handleClick}
        {...props}
      >
        {children}
      </div>
    </PromptInputContext.Provider>
  )
}

export type PromptInputTextareaProps = {
  disableAutosize?: boolean
} & React.ComponentProps<"textarea">

function PromptInputTextarea({
  className,
  onKeyDown,
  disableAutosize = false,
  ...props
}: PromptInputTextareaProps) {
  const { value, setValue, maxHeight, onSubmit, disabled, textareaRef } = usePromptInput()

  useLayoutEffect(() => {
    if (!textareaRef.current || disableAutosize) return
    const el = textareaRef.current
    el.style.height = "auto"
    if (typeof maxHeight === "number") {
      el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`
    } else {
      el.style.height = `min(${el.scrollHeight}px, ${maxHeight})`
    }
  }, [value, maxHeight, disableAutosize, textareaRef])

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onSubmit?.()
    }
    onKeyDown?.(e)
  }

  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={handleChange}
      onKeyDown={handleKeyDown}
      disabled={disabled}
      rows={1}
      className={cn(
        "w-full resize-none border-0 bg-transparent text-sm text-white placeholder:text-white/30 focus:outline-none disabled:cursor-not-allowed",
        className
      )}
      {...props}
    />
  )
}

export type PromptInputActionsProps = React.HTMLAttributes<HTMLDivElement>

function PromptInputActions({ children, className, ...props }: PromptInputActionsProps) {
  return (
    <div className={cn("flex items-center justify-between gap-2 pt-2", className)} {...props}>
      {children}
    </div>
  )
}

export type PromptInputActionProps = {
  className?: string
  tooltip?: React.ReactNode
  children: React.ReactNode
} & React.ComponentProps<"button">

function PromptInputAction({ tooltip, children, className, ...props }: PromptInputActionProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-lg p-2 text-white/40 transition-colors hover:bg-white/10 hover:text-white/70 disabled:cursor-not-allowed disabled:opacity-40",
        className
      )}
      title={typeof tooltip === "string" ? tooltip : undefined}
      {...props}
    >
      {children}
    </button>
  )
}

export { PromptInput, PromptInputTextarea, PromptInputActions, PromptInputAction }
