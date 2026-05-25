import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const spawnMock = vi.fn()
const existsSyncMock = vi.fn()

vi.mock('node:child_process', () => ({
  spawn: spawnMock
}))

vi.mock('node:fs', async importOriginal => {
  const actual = await importOriginal<typeof import('node:fs')>()

  return {
    ...actual,
    existsSync: existsSyncMock
  }
})

class FakeChild extends EventEmitter {
  exitCode: null | number = null
  killed = false
  stderr = new PassThrough()
  stdin = new PassThrough()
  stdout = new PassThrough()

  kill() {
    this.killed = true
    this.exitCode = 0
    this.emit('exit', 0)
    return true
  }
}

describe('GatewayClient backend selection', () => {
  const envBackup = { ...process.env }

  beforeEach(() => {
    vi.resetModules()
    spawnMock.mockReset()
    existsSyncMock.mockReset()
    process.env = {
      ...envBackup,
      HERMES_PYTHON_SRC_ROOT: '/repo',
      HERMES_CWD: '/repo',
      HERMES_PYTHON: '/definitely/missing/python'
    }
    existsSyncMock.mockImplementation(path => String(path) === '/repo/Cargo.toml')
    spawnMock.mockImplementation(() => new FakeChild())
  })

  afterEach(() => {
    process.env = { ...envBackup }
  })

  it('prefers the Rust cargo backend by default even when HERMES_PYTHON is invalid', async () => {
    const { GatewayClient } = await import('../gatewayClient.js')
    const client = new GatewayClient()

    client.start()
    client.kill()

    expect(spawnMock).toHaveBeenCalledWith(
      'cargo',
      ['run', '-q', '-p', 'hermes-rs-cli', '--bin', 'hermes', '--', 'tui-gateway'],
      expect.objectContaining({ cwd: '/repo' })
    )
  })

  it('uses the explicit gateway binary from the Rust launcher env', async () => {
    process.env.HERMES_TUI_GATEWAY_BIN = '/tmp/hermes'
    process.env.HERMES_TUI_GATEWAY_ARGS_JSON = '["tui-gateway"]'

    const { GatewayClient } = await import('../gatewayClient.js')
    const client = new GatewayClient()

    client.start()
    client.kill()

    expect(spawnMock).toHaveBeenCalledWith(
      '/tmp/hermes',
      ['tui-gateway'],
      expect.objectContaining({ cwd: '/repo' })
    )
  })
})
