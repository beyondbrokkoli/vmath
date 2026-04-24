local ffi = require("ffi")
local bit = require("bit")
local VibeMath = require("load")
local Memory = require("memory")

-- Inject the Camera struct into your camera module
local Camera = require("camera")(Memory.Camera)

