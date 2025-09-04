// NilorPreset dynamic inputs extension
// Adds a new empty input slot whenever the last slot gets connected, up to a hard cap

;(function () {
  const MAX_INPUTS = 32
  const CLASS_TYPE = 'NilorPreset'

  function isNilorPreset(node) {
    return node && (node.type === CLASS_TYPE || node?.comfyClass === CLASS_TYPE)
  }

  function nextInputName(node) {
    let index = 1
    while (index <= MAX_INPUTS) {
      const key = `input_${index}`
      if (!node.inputs || !node.inputs.find((i) => i.name === key)) {
        return key
      }
      index++
    }
    return null
  }

  function countInputs(node) {
    return (node.inputs || []).filter((i) => i && i.name?.startsWith('input_')).length
  }

  function ensureAtLeastOneSlot(node) {
    if (!isNilorPreset(node)) return
    if (countInputs(node) === 0) {
      const name = 'input_1'
      node.addInput(name, 'PRESET_HOOK')
      node.updateNode?.()
    }
  }

  function addAnotherSlotIfLastWasLinked(node) {
    if (!isNilorPreset(node)) return
    const inputs = (node.inputs || []).filter((i) => i && i.name?.startsWith('input_'))
    if (inputs.length === 0) return
    const last = inputs[inputs.length - 1]
    const lastIsLinked = !!last.link
    if (lastIsLinked && inputs.length < MAX_INPUTS) {
      const name = nextInputName(node)
      if (name) {
        node.addInput(name, 'PRESET_HOOK')
        node.updateNode?.()
      }
    }
  }

  function onNodeCreated(node) {
    if (!isNilorPreset(node)) return
    ensureAtLeastOneSlot(node)
    addAnotherSlotIfLastWasLinked(node)
  }

  function onNodeConnected(linkInfo) {
    const node = linkInfo?.target?.node
    if (!node) return
    addAnotherSlotIfLastWasLinked(node)
  }

  function onNodeDisconnected(linkInfo) {
    const node = linkInfo?.target?.node
    if (!node) return
    // No shrinking logic for V1; keep existing inputs
  }

  if (typeof ComfyUI !== 'undefined' && ComfyUI?.registerExtension) {
    ComfyUI.registerExtension({
      name: 'nilor-preset-dynamic-inputs',
      nodeCreated(node) {
        onNodeCreated(node)
      },
      afterConfigureGraph(graph) {
        // Ensure existing NilorPreset nodes in loaded graphs are initialized
        try {
          ;(graph?._nodes || graph?.nodes || []).forEach((n) => onNodeCreated(n))
        } catch (e) {
          console.warn('nilor-preset-dynamic-inputs init error', e)
        }
      },
      nodeConnected(edge) {
        try {
          onNodeConnected(edge)
        } catch (e) {
          console.warn('nilor-preset-dynamic-inputs nodeConnected error', e)
        }
      },
      nodeDisconnected(edge) {
        try {
          onNodeDisconnected(edge)
        } catch (e) {
          console.warn('nilor-preset-dynamic-inputs nodeDisconnected error', e)
        }
      },
    })
  }
})()


