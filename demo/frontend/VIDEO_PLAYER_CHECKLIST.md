# SAM2 Video Player Implementation Checklist

## Initial Setup
- [ ] Install required dependencies
  - [ ] React and TypeScript
    ```typescript
    type VideoRef = {
      getCanvas(): HTMLCanvasElement | null;
      width: number;
      height: number;
      frame: number;
      numberOfFrames: number;
      // ... other interface methods
    }
    ```
  - [ ] StyleX for styling
    ```typescript
    const styles = stylex.create({
      container: {position: 'relative', width: '100%', height: '100%'},
      canvasContainer: {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: color['gray-800'],
      }
    });
    ```
  - [ ] Carbon icons (PauseFilled, PlayFilledAlt)
  - [ ] React DaisyUI for UI components
  - [ ] Jotai for state management
    ```typescript
    const [isPlaying, setIsPlaying] = useAtom(isPlayingAtom);
    const [isVideoLoading, setIsVideoLoading] = useAtom(isVideoLoadingAtom);
    ```

## Core Architecture

### 1. Worker Communication Layer
- [ ] VideoWorkerBridge Implementation
  ```typescript
  class VideoWorkerBridge extends EventEmitter<VideoWorkerEventMap> {
    protected worker: Worker;
    private metadata: Metadata | null = null;
    private frameIndex: number = 0;
    private _sessionId: string | null = null;
  }
  ```
- [ ] Message Handling
  - [ ] Request Types
  ```typescript
  type VideoWorkerRequest = 
    | SetCanvasRequest
    | SetSourceRequest 
    | PlayRequest
    | PauseRequest
    | FrameUpdateRequest;
  ```
  - [ ] Response Types
  ```typescript
  type VideoWorkerResponse =
    | ErrorResponse
    | DecodeResponse
    | PlayResponse
    | PauseResponse
    | FrameUpdateResponse;
  ```

### 2. Context Management
- [ ] VideoWorkerContext Core
  ```typescript
  class VideoWorkerContext {
    private _canvas: OffscreenCanvas | null = null;
    private _ctx: OffscreenCanvasRenderingContext2D | null = null;
    private _form: CanvasForm | null = null;
    private _decodedVideo: DecodedVideo | null = null;
    private _frameIndex: number = 0;
    private _isPlaying: boolean = false;
  }
  ```
- [ ] WebGL Context Management
  ```typescript
  private initializeWebGLContext(width: number, height: number): void {
    this._canvasHighlights = new OffscreenCanvas(width, height);
    this._glObjects = this._canvasHighlights.getContext('webgl2');
    this._canvasBackground = new OffscreenCanvas(width, height);
    this._glBackground = this._canvasBackground.getContext('webgl2');
  }
  ```

## Video Processing Pipeline

### 1. Decoder Implementation
- [ ] Stream Setup
  ```typescript
  private async _decodeVideo(src: string): Promise<void> {
    const fileStream = streamFile(src, {
      credentials: 'same-origin',
      cache: 'no-store'
    });
    this._decodedVideo = await decodeStream(fileStream, progress => {
      // Process decoded frames
    });
  }
  ```

### 2. Frame Management
- [ ] Frame State Tracking
  ```typescript
  private _frameIndex: number = 0;
  private _decodedVideo: DecodedVideo | null = null;
  private _isDrawing: boolean = false;
  ```
- [ ] Frame Navigation
  ```typescript
  public goToFrame(index: number): void {
    this._cancelRender();
    this.updateFrameIndex(index);
    this._playbackRAFHandle = requestAnimationFrame(this._drawFrame.bind(this));
  }
  ```

### 3. Rendering Pipeline
- [ ] Frame Drawing
  ```typescript
  private async _drawFrameImpl(
    form: CanvasForm,
    frameIndex: number,
    enableWatermark: boolean = false,
    step: number = 0,
    maxSteps: number = 40
  ): Promise<void> {
    const frame = this._decodedVideo.frames[frameIndex];
    const frameBitmap = await createImageBitmap(frame.bitmap);
    // Apply effects and render
  }
  ```

## Effect System

### 1. Effect Management
- [ ] Effect Registration
  ```typescript
  constructor() {
    this._effects = [
      AllEffects.Original,  // Background layer
      AllEffects.Overlay,   // Mask overlay layer
    ];
  }
  ```
- [ ] Effect Processing
  ```typescript
  private _processEffects(
    form: CanvasForm,
    effectParams: EffectFrameContext,
    tracklets: Tracklet[]
  ) {
    for (let i = 0; i < this._effects.length; i++) {
      const effect = this._effects[i];
      effect.apply(form, effectParams, tracklets);
    }
  }
  ```

## Playback Control

### 1. Playback State Management
- [ ] State Variables
  ```typescript
  private _isPlaying: boolean = false;
  private _playbackRAFHandle: number | null = null;
  private _playbackTimeoutHandle: NodeJS.Timeout | null = null;
  ```
- [ ] Playback Control
  ```typescript
  public play(): void {
    if (this._isPlaying) return;
    const {numFrames, fps} = this._decodedVideo;
    const timePerFrame = 1000 / (fps ?? 30);
    this.updatePlayback(true);
  }
  ```

### 2. Animation Control
- [ ] Animation Frame Management
  ```typescript
  private _cancelRender(): void {
    if (this._playbackRAFHandle !== null) {
      cancelAnimationFrame(this._playbackRAFHandle);
      this._playbackRAFHandle = null;
    }
  }
  ```

## Performance Optimization

### 1. Memory Management
- [ ] Resource Cleanup
  ```typescript
  public close(): void {
    this._ctx?.reset();
    this._decodedVideo?.frames.forEach(f => f.bitmap.close());
    this._decodedVideo = null;
  }
  ```
- [ ] Frame Buffer Management
  ```typescript
  private async *_framesGenerator(
    decodedVideo: DecodedVideo,
    canvas: OffscreenCanvas,
    form: CanvasForm,
  ): AsyncGenerator<ImageFrame, undefined> {
    // Generate frames with proper cleanup
  }
  ```

### 2. Performance Monitoring
- [ ] Statistics Tracking
  ```typescript
  public enableStats() {
    this._stats.fps = new Stats('fps');
    this._stats.videoFps = new Stats('fps', 'V');
    this._stats.total = new Stats('ms', 'T');
    this._stats.effect0 = new Stats('ms', 'B');
    this._stats.effect1 = new Stats('ms', 'H');
  }
  ```

## Event System

### 1. Event Types
```typescript
interface VideoWorkerEventMap {
  error: ErrorEvent;
  decode: DecodeEvent;
  play: PlayEvent;
  pause: PauseEvent;
  frameUpdate: FrameUpdateEvent;
  streamingStateUpdate: StreamingStateUpdateEvent;
}
```

### 2. Event Management
- [ ] Event Registration
  ```typescript
  constructor(worker: Worker) {
    worker.addEventListener('message', (event) => {
      switch (event.data.action) {
        case 'decode': this.metadata = event.data; break;
        case 'frameUpdate': this.frameIndex = event.data.index; break;
      }
      this.trigger(event.data.action, event.data);
    });
  }
  ```

## Error Handling

### 1. Error Processing
- [ ] Error Serialization
  ```typescript
  private _sendRenderingError(error: Error): void {
    const serializedError = serializeError(error);
    this.sendResponse<RenderingErrorResponse>('renderingError', {
      error: serializedError
    });
  }
  ```

### 2. Recovery Mechanisms
- [ ] WebGL Context Loss
  ```typescript
  this._canvasHighlights.addEventListener(
    'webglcontextlost',
    event => {
      event.preventDefault();
      this._sendRenderingError(new WebGLContextError('WebGL context lost.'));
    },
    false
  );
  ```

## Browser Compatibility

### 1. Safari-Specific Handling
- [ ] Visibility Change
  ```typescript
  function onVisibilityChange() {
    if (document.visibilityState === 'visible' && !isPlaying) {
      bridge.goToFrame(bridge.frame);
    }
  }
  ```
- [ ] Frame Bitmap Conversion
  ```typescript
  // Convert VideoFrame to ImageBitmap for Safari compatibility
  const frameBitmap = await createImageBitmap(bitmap);
  ```

### 2. WebGL Context Management
```typescript
private initializeWebGLContext(width: number, height: number): void {
  if (this._canvasHighlights == null && this._glObjects == null) {
    this._canvasHighlights = new OffscreenCanvas(width, height);
    this._glObjects = this._canvasHighlights.getContext('webgl2');
  }
}
```

## Resource Management

### 1. Cleanup Procedures
```typescript
public terminate(): void {
  super.destroy();
  this.worker.terminate();
}
```

### 2. Canvas Management
```typescript
public setCanvas(canvas: OffscreenCanvas) {
  this._canvas = canvas;
  this._ctx = canvas.getContext('2d');
  this._form = new CanvasForm(this._ctx);
}
```

## Security Measures

### 1. Resource Access
```typescript
const fileStream = streamFile(src, {
  credentials: 'same-origin',
  cache: 'no-store',
});
```

### 2. Worker Security
```typescript
// Careful exposure of worker instance
getWorker_ONLY_USE_WITH_CAUTION(): Worker {
  return this.worker;
}
```

## Testing Strategy

### 1. Unit Testing
- [ ] Worker Communication Tests
  ```typescript
  describe('VideoWorkerBridge', () => {
    test('should handle decode events', async () => {
      const bridge = new VideoWorkerBridge();
      const mockEvent = new MessageEvent('message', {
        data: { action: 'decode', metadata: {...} }
      });
      await expect(bridge.handleMessage(mockEvent)).resolves.toBeDefined();
    });
  });
  ```

### 2. Integration Testing
- [ ] Video Processing Pipeline
  ```typescript
  describe('Video Processing', () => {
    test('should process video frames correctly', async () => {
      const context = new VideoWorkerContext();
      await context.setSource('test.mp4');
      expect(context.numberOfFrames).toBeGreaterThan(0);
    });
  });
  ```

### 3. Performance Testing
- [ ] Frame Rate Benchmarks
  ```typescript
  test('should maintain target frame rate', async () => {
    const player = new VideoPlayer();
    const frameTimings = await measureFrameTimings(player);
    expect(calculateAverageFPS(frameTimings)).toBeGreaterThanOrEqual(30);
  });
  ```

## Accessibility Implementation

### 1. Keyboard Controls
- [ ] Keyboard Navigation
  ```typescript
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      switch(event.key) {
        case 'Space':
          togglePlayPause();
          break;
        case 'ArrowRight':
          skipForward();
          break;
        case 'ArrowLeft':
          skipBackward();
          break;
      }
    };
    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, []);
  ```

### 2. ARIA Attributes
- [ ] Control Labels
  ```typescript
  <button
    aria-label={isPlaying ? 'Pause' : 'Play'}
    onClick={togglePlayPause}
    role="button"
    tabIndex={0}
  >
    {isPlaying ? <PauseIcon /> : <PlayIcon />}
  </button>
  ```

### 3. Screen Reader Support
- [ ] Progress Announcements
  ```typescript
  const announceProgress = (frameIndex: number) => {
    const progress = Math.round((frameIndex / totalFrames) * 100);
    announceToScreenReader(`Video progress: ${progress}%`);
  };
  ```

## Mobile Device Compatibility

### 1. Touch Controls
- [ ] Gesture Handling
  ```typescript
  const handleTouchGesture = (event: TouchEvent) => {
    if (event.touches.length === 2) {
      handlePinchZoom(event);
    } else {
      handleSwipe(event);
    }
  };
  ```

### 2. Responsive Layout
- [ ] Viewport Adaptation
  ```typescript
  const styles = stylex.create({
    videoContainer: {
      width: '100%',
      height: 'auto',
      maxHeight: '100vh',
      '@media (max-width: 768px)': {
        height: '50vh',
      },
    },
  });
  ```

### 3. Performance Optimization for Mobile
- [ ] Resource Management
  ```typescript
  const optimizeForMobile = () => {
    if (isMobileDevice()) {
      setMaxBufferedFrames(Math.min(30, maxBufferedFrames));
      setRenderQuality('medium');
    }
  };
  ```

## Documentation

### 1. API Documentation
- [ ] Public Methods
  ```typescript
  /**
   * Controls video playback state
   * @param {boolean} shouldPlay - Whether to play or pause
   * @returns {Promise<void>} Resolves when playback state changes
   */
  public async setPlaying(shouldPlay: boolean): Promise<void>
  ```

### 2. Usage Examples
- [ ] Basic Implementation
  ```typescript
  const VideoPlayerExample = () => {
    return (
      <Video
        src="example.mp4"
        autoPlay={false}
        controls={true}
        onLoad={() => console.log('Video loaded')}
      />
    );
  };
  ```

## Performance Benchmarking

### 1. Frame Rate Analysis
- [ ] Metrics Collection
  ```typescript
  interface PerformanceMetrics {
    averageFPS: number;
    frameDrops: number;
    renderTime: number;
    memoryUsage: number;
  }
  ```

### 2. Memory Profiling
- [ ] Memory Leak Detection
  ```typescript
  const monitorMemoryUsage = () => {
    if (performance.memory) {
      const {usedJSHeapSize, totalJSHeapSize} = performance.memory;
      logMemoryMetrics(usedJSHeapSize, totalJSHeapSize);
    }
  };
  ```

## External Dependencies

### 1. Package Management
```json
{
  "dependencies": {
    "@carbon/icons-react": "^11.17.0",
    "jotai": "^2.0.0",
    "react": "^18.2.0",
    "react-use": "^17.4.0",
    "stylex": "^0.3.0"
  }
}
```

### 2. Browser Support Matrix
```typescript
const SUPPORTED_BROWSERS = {
  chrome: '>= 80',
  firefox: '>= 75',
  safari: '>= 13',
  edge: '>= 80'
};
```

## Development Workflow

### 1. Build Process
- [ ] Development Build
  ```bash
  npm run dev
  ```
- [ ] Production Build
  ```bash
  npm run build
  ```

### 2. Code Quality
- [ ] Linting Configuration
  ```json
  {
    "extends": [
      "eslint:recommended",
      "plugin:@typescript-eslint/recommended"
    ],
    "rules": {
      "@typescript-eslint/explicit-function-return-type": "error"
    }
  }
  ```

## Advanced Features

### 1. Video Seek Optimization
- [ ] Frame Indexing
  ```typescript
  interface FrameIndex {
    timestamp: number;
    offset: number;
    keyframe: boolean;
    metadata: {
      resolution: { width: number; height: number };
      codec: string;
    };
  }
  ```
- [ ] Seek Implementation
  ```typescript
  class VideoSeeker {
    private frameIndices: FrameIndex[] = [];
    private keyframeInterval: number = 30;
    
    async seekToTime(timestamp: number): Promise<void> {
      const nearestKeyframe = this.findNearestKeyframe(timestamp);
      await this.decodeFromKeyframe(nearestKeyframe);
      this.renderFrame(timestamp);
    }
  }
  ```

### 2. Quality Adaptation
- [ ] Dynamic Resolution Scaling
  ```typescript
  class QualityManager {
    private readonly targetFPS = 30;
    private qualityLevels = ['high', 'medium', 'low'] as const;
    
    adjustQuality(currentFPS: number): void {
      if (currentFPS < this.targetFPS * 0.8) {
        this.downscaleResolution();
      } else if (currentFPS > this.targetFPS * 1.2) {
        this.upscaleResolution();
      }
    }
  }
  ```

### 3. Memory Management
- [ ] Frame Caching Strategy
  ```typescript
  class FrameCache {
    private cache: Map<number, ImageBitmap> = new Map();
    private readonly maxCacheSize = 100;
    
    add(frameIndex: number, frame: ImageBitmap): void {
      if (this.cache.size >= this.maxCacheSize) {
        const oldestFrame = this.cache.keys().next().value;
        this.cache.get(oldestFrame)?.close();
        this.cache.delete(oldestFrame);
      }
      this.cache.set(frameIndex, frame);
    }
  }
  ```

## Enhanced Error Handling

### 1. Decoder Error Recovery
- [ ] Error Classification
  ```typescript
  enum DecoderErrorType {
    MEDIA_NOT_SUPPORTED = 'MEDIA_NOT_SUPPORTED',
    NETWORK_ERROR = 'NETWORK_ERROR',
    DECODE_ERROR = 'DECODE_ERROR',
    OUT_OF_MEMORY = 'OUT_OF_MEMORY'
  }
  ```
- [ ] Recovery Strategies
  ```typescript
  class DecoderErrorHandler {
    async handleError(error: DecoderError): Promise<void> {
      switch (error.type) {
        case DecoderErrorType.NETWORK_ERROR:
          await this.retryWithBackoff();
          break;
        case DecoderErrorType.OUT_OF_MEMORY:
          await this.releaseMemoryAndRetry();
          break;
        case DecoderErrorType.MEDIA_NOT_SUPPORTED:
          await this.fallbackToSoftwareDecoder();
          break;
      }
    }
  }
  ```

## WebGL Optimization

### 1. Shader Programs
- [ ] Custom Shaders
  ```glsl
  // Vertex Shader
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0, 1);
    v_texCoord = a_texCoord;
  }
  
  // Fragment Shader
  precision mediump float;
  uniform sampler2D u_image;
  varying vec2 v_texCoord;
  
  void main() {
    gl_FragColor = texture2D(u_image, v_texCoord);
  }
  ```

### 2. Performance Optimization
- [ ] WebGL State Management
  ```typescript
  class GLStateManager {
    private readonly gl: WebGL2RenderingContext;
    private activeTexture: number = -1;
    private boundProgram: WebGLProgram | null = null;
    
    setActiveTexture(unit: number): void {
      if (this.activeTexture !== unit) {
        this.gl.activeTexture(this.gl.TEXTURE0 + unit);
        this.activeTexture = unit;
      }
    }
  }
  ```

## Advanced Testing

### 1. Performance Testing Suite
- [ ] Frame Time Analysis
  ```typescript
  class FrameTimeAnalyzer {
    private frameTimes: number[] = [];
    private readonly sampleSize = 60;
    
    addFrameTime(time: number): void {
      this.frameTimes.push(time);
      if (this.frameTimes.length > this.sampleSize) {
        this.frameTimes.shift();
      }
    }
    
    getMetrics(): FrameMetrics {
      return {
        average: this.calculateAverage(),
        percentile99: this.calculatePercentile(99),
        jank: this.calculateJank()
      };
    }
  }
  ```

### 2. Memory Leak Detection
- [ ] Automated Memory Analysis
  ```typescript
  class MemoryLeakDetector {
    private snapshots: MemorySnapshot[] = [];
    
    async captureSnapshot(): Promise<void> {
      if ('performance' in window && 'memory' in performance) {
        const { usedJSHeapSize, totalJSHeapSize } = performance.memory;
        this.snapshots.push({
          timestamp: Date.now(),
          usedHeap: usedJSHeapSize,
          totalHeap: totalJSHeapSize
        });
      }
    }
    
    analyzeGrowth(): MemoryGrowthMetrics {
      // Analyze memory growth patterns
      return this.calculateGrowthMetrics();
    }
  }
  ```

## Security Enhancements

### 1. Content Security
- [ ] Source Validation
  ```typescript
  class ContentSecurityValidator {
    private readonly allowedOrigins: string[] = [];
    private readonly maxFileSize = 100 * 1024 * 1024; // 100MB
    
    async validateSource(src: string): Promise<boolean> {
      const origin = new URL(src).origin;
      if (!this.allowedOrigins.includes(origin)) {
        throw new SecurityError('Invalid content origin');
      }
      
      const headers = await this.fetchHeaders(src);
      return this.validateHeaders(headers);
    }
  }
  ```

### 2. Worker Security
- [ ] Message Validation
  ```typescript
  class WorkerMessageValidator {
    private readonly allowedMessageTypes = new Set([
      'decode',
      'play',
      'pause',
      'seek'
    ]);
    
    validateMessage(message: unknown): boolean {
      if (!this.isValidMessageFormat(message)) {
        return false;
      }
      
      return this.allowedMessageTypes.has(message.type);
    }
  }
  ```

## Mobile Optimizations

### 1. Battery Awareness
- [ ] Power Management
  ```typescript
  class PowerManager {
    private batteryLevel: number = 1;
    
    async initialize(): Promise<void> {
      const battery = await navigator.getBattery();
      this.batteryLevel = battery.level;
      
      battery.addEventListener('levelchange', () => {
        this.batteryLevel = battery.level;
        this.adjustPerformance();
      });
    }
    
    private adjustPerformance(): void {
      if (this.batteryLevel < 0.2) {
        this.enablePowerSaving();
      }
    }
  }
  ```

### 2. Network Awareness
- [ ] Adaptive Streaming
  ```typescript
  class NetworkAwareLoader {
    private connection: NetworkInformation;
    
    constructor() {
      this.connection = navigator.connection;
      this.connection.addEventListener('change', 
        this.handleConnectionChange.bind(this));
    }
    
    private handleConnectionChange(): void {
      const effectiveType = this.connection.effectiveType;
      this.adjustQualityForNetwork(effectiveType);
    }
  }
  ```

## Analytics Integration

### 1. Performance Monitoring
- [ ] Metrics Collection
  ```typescript
  class AnalyticsCollector {
    private readonly metrics: MetricsBuffer = {
      playback: new CircularBuffer<PlaybackMetric>(100),
      performance: new CircularBuffer<PerformanceMetric>(100),
      errors: new CircularBuffer<ErrorMetric>(50)
    };
    
    collectMetric(type: MetricType, data: any): void {
      this.metrics[type].push({
        timestamp: Date.now(),
        ...data
      });
    }
    
    flushMetrics(): Promise<void> {
      return this.sendToAnalyticsService(this.metrics);
    }
  }
  ```

### 2. User Interaction Tracking
- [ ] Event Logging
  ```typescript
  class InteractionTracker {
    private readonly interactions: UserInteraction[] = [];
    
    trackInteraction(type: InteractionType, data: any): void {
      this.interactions.push({
        type,
        timestamp: Date.now(),
        data
      });
    }
  }
  ```
