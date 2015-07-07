#ifndef SAMPLE_FREQ_PASS
#define PIXEL_FREQ_PASS
#endif

#ifndef NUMTHREADS_X
#define NUMTHREADS_X NUMTHREADS
#endif

#ifndef NUMTHREADS_Y
#define NUMTHREADS_Y NUMTHREADS_X
#endif

#define GROUP_THREADS NUMTHREADS_X * NUMTHREADS_Y

#include <frame.h>
#include <csm.h>
#include <math.h>


Texture2D<float>	DepthBuffer	: register(t0);

#ifndef MS_SAMPLE_COUNT
Texture2D<uint2>	Stencil		: register(t1);
#else
Texture2DMS<uint2, MS_SAMPLE_COUNT>	StencilMS	: register(t1);
#endif


Texture2D<float> Shadow : register(t0);
RWTexture2D<float> Output : register(u0);

//

static const float2 PoissonSamplesArray[] = {
	float2(0.130697, -0.209628),
	float2(-0.112312, 0.327448),
	float2(-0.499089, -0.030236),
	float2(0.332994, 0.380106),
	float2(-0.234209, -0.557516),
	float2(0.695785, 0.066096),
	float2(-0.419485, 0.632050),
	float2(0.678688, -0.447710),
	float2(0.333877, 0.807633),
	float2(-0.834613, 0.383171),
	float2(-0.682884, -0.637443),
	float2(0.769794, 0.568801),
	float2(-0.087941, -0.955035),
	float2(-0.947188, -0.166568),
	float2(0.425303, -0.874130),
	float2(-0.134360, 0.982611),
};

static const float2 RandomRotation[] = {
	float2(0.971327, 0.237749),
	float2(-0.885968, 0.463746),
	float2(-0.913331, 0.407218),
	float2(0.159352, 0.987222),
	float2(-0.640909, 0.767617),
	float2(-0.625570, 0.780168),
	float2(-0.930406, 0.366530),
	float2(-0.940038, 0.341070),
	float2(0.964899, 0.262621),
	float2(-0.647723, 0.761876),
	float2(0.663773, 0.747934),
	float2(0.929892, 0.367833),
	float2(-0.686272, 0.727345),
	float2(-0.999057, 0.043413),
	float2(-0.710684, 0.703511),
	float2(-0.893640, 0.448784)
};

static const int2 SampleOffset[] = {
	int2(-1, 0),
	int2(-1, -1),
	int2(0, -1),
	int2(1, -1),
	int2(1, 0),
	int2(1, 1),
	int2(0, 1),
	int2(-1, 1),
	int2(-2, 1),
	int2(-2, 0),
	int2(-2, -1),
	int2(-2, -2),
	int2(-1, -2),
	int2(0, -2),
	int2(1, -2),
	int2(2, -2),
	int2(2, -1),
	int2(2, 0),
	int2(2, 1),
	int2(2, 2),
	int2(1, 2),
	int2(0, 2),
	int2(-1, 2),
	int2(-2, 2)
};

static const uint PoissonSamplesNum = 16;
static const float FilterSize = 3;

[numthreads(NUMTHREADS_X, NUMTHREADS_Y, 1)]
void write_shadow(
	uint3 dispatchThreadID : SV_DispatchThreadID,
	uint3 groupThreadID : SV_GroupThreadID,
	uint3 GroupID : SV_GroupID,
	uint ThreadIndex : SV_GroupIndex) {

	float2 Texel = dispatchThreadID.xy;

		float2 uv = (Texel + 0.5f) / frame_.resolution;
		float3 pos = reconstruct_position(DepthBuffer[Texel], uv);

#ifndef MS_SAMPLE_COUNT
		uint c_id = cascade_id_stencil(Stencil[Texel].y);
#else
		uint c_id = cascade_id_stencil(StencilMS.Load(Texel, 0).y);
#endif

	float3 lpos = world_to_shadowmap(pos, csm_.cascade_matrix[c_id]);

		float texelsize = 1 / 512.f;
	float2 filterSize = csm_.cascade_scale[c_id].xy * FilterSize;

		float result = 0;
	[branch]
	if (filterSize.x > 1.0f || filterSize.y > 1.0f) {

		uint2 rotationOffset = dispatchThreadID.xy % 4;
			float2 theta = RandomRotation[rotationOffset.x * 4 + rotationOffset.y].xy;
			float2x2 rotMatrix = float2x2(float2(theta.xy), float2(-theta.y, theta.x));

			[unroll]
		for (uint i = 0; i<PoissonSamplesNum; i++) {
			float2 offset = filterSize * 0.5f * PoissonSamplesArray[i] * texelsize;
				offset = mul(rotMatrix, offset);
			result += CSM.SampleCmpLevelZero(ShadowmapSampler, float3(lpos.xy + offset, c_id), lpos.z);
		}
		result /= PoissonSamplesNum;
	}
	else {
		result = CSM.SampleCmpLevelZero(ShadowmapSampler, float3(lpos.xy, c_id), lpos.z) + any(saturate(lpos.xy) != (lpos.xy));
	}

	Output[Texel] = result;
}

float shadow(float2 offset, int2 texeloffset, float scalexy, float scalez, float distancez)
{
	float sunh = 0.005f;
	float htexelsize = scalexy / 1.9f;//in m

	//if (distancez < 0.02f)
	//distancez = 0.02f;//in 0 to 1
	float3 distance = float3((offset + texeloffset) * scalexy, distancez * scalez);//in m

		float sh = sunh * distance.z;//in m
	float j = sh - htexelsize;//in m

	float2 light = saturate((distance.xy + j) / (sh * 2)) + saturate((j - distance.xy) / (sh * 2));

		return (1 - light.x) * (1 - light.y);
	//result += 1 - saturate(light.x + light.y);
	//result += 1 - max(light.x, light.y);
}

[numthreads(NUMTHREADS_X, NUMTHREADS_Y, 1)]
void write_shadow_pcss(
	uint3 dispatchThreadID : SV_DispatchThreadID,
	uint3 groupThreadID : SV_GroupThreadID,
	uint3 GroupID : SV_GroupID,
	uint ThreadIndex : SV_GroupIndex) {

	float2 Texel = dispatchThreadID.xy;

		float2 uv = (Texel + 0.5f) / frame_.resolution;
		float3 pos = reconstruct_position(DepthBuffer[Texel], uv);

#ifndef MS_SAMPLE_COUNT
		uint c_id = cascade_id_stencil(Stencil[Texel].y);
#else
		uint c_id = cascade_id_stencil(StencilMS.Load(Texel, 0).y);
#endif

	float sunh = 0.005f;
	float result = 0;


	float3 lpos = world_to_shadowmap(pos, csm_.cascade_matrix[3]);
	float scalez = 136.0f / csm_.cascade_scale[3].z;//0 to 1 to m for z
	float scalexy = 36.0f / (csm_.cascade_scale[3].x * 1024.f);//t to m for xy
	int2 texel = lpos.xy * 1024;//in t
	float2 offset = texel - (lpos.xy * 1024.f) + float2(0.5f, 0.5f);//in t
	
	float zpos = 0.0f;
	float distancez = 0.0f;

	[unroll]
	for (uint i = 0; i < 24; i++)//cascade 3
	{
		//int2 texeloffset = int2(i / 5 - 2, i % 5 - 2);//in t
		int2 texeloffset = SampleOffset[i].xy;//in t
		zpos = CSM.Load(int4(texel, 3, 0), texeloffset);//in 0 to 1
		distancez = lpos.z - zpos;//in 0 to 1

		if (distancez >= 0.1f)
		{
			result += shadow(offset, texeloffset, scalexy, scalez, distancez);
		}
	}
	
	zpos = CSM.Load(int4(texel, 3, 0));//in 0 to 1
	distancez = lpos.z - zpos;//in 0 to 1

	if (c_id < 3)//cascade 2
	{
		lpos = world_to_shadowmap(pos, csm_.cascade_matrix[2]);
		scalez = 136.0f / csm_.cascade_scale[2].z;//0 to 1 to m for z
		scalexy = 36.0f / (csm_.cascade_scale[2].x * 1024.f);//t to m for xy
		texel = lpos.xy * 1024 + float2(0.5f, 0.5f);//in t
		offset = texel - (lpos.xy * 1024.f) + float2(0.5f, 0.5f);//in t

		[unroll]
		for (uint i = 0; i < 15; i++)
		{
			//int2 texeloffset = int2(i / 4 - 2, i % 4 - 2);//in t
			int2 texeloffset = SampleOffset[i].xy;//in t
			zpos = CSM.Load(int4(texel, 2, 0), texeloffset);//in 0 to 1
			distancez = lpos.z - zpos;//in 0 to 1

			if (distancez >= 0.1f)
			{
				result += shadow(offset, texeloffset, scalexy, scalez, distancez);
			}
		}

		zpos = CSM.Load(int4(texel, 2, 0));//in 0 to 1
		distancez = lpos.z - zpos;//in 0 to 1

		if (c_id < 2)//cascade 1
		{
			lpos = world_to_shadowmap(pos, csm_.cascade_matrix[1]);
			scalez = 136.0f / csm_.cascade_scale[1].z;//0 to 1 to m for z
			scalexy = 36.0f / (csm_.cascade_scale[1].x * 1024.f);//t to m for xy
			texel = lpos.xy * 1024;//in t
			offset = texel - (lpos.xy * 1024.f) + float2(0.5f, 0.5f);//in t

			[unroll]
			for (uint i = 0; i < 24; i++)
			{
				//int2 texeloffset = int2(i / 5 - 2, i % 5 - 2);//in t
				int2 texeloffset = SampleOffset[i].xy;//in t
				zpos = CSM.Load(int4(texel, 1, 0), texeloffset);//in 0 to 1
				distancez = lpos.z - zpos;//in 0 to 1

				if (distancez >= 0.1f)
				{
					result += shadow(offset, texeloffset, scalexy, scalez, distancez);
				}
			}

			zpos = CSM.Load(int4(texel, 1, 0));//in 0 to 1
			distancez = lpos.z - zpos;//in 0 to 1

			if (c_id < 1)//cascade 0
			{
				lpos = world_to_shadowmap(pos, csm_.cascade_matrix[0]);
				scalez = 136.0f / csm_.cascade_scale[0].z;//0 to 1 to m for z
				scalexy = 36.0f / (csm_.cascade_scale[0].x * 1024.f);//t to m for xy
				texel = (lpos.xy * 1024) / 3.0f + float2(0.5f, 0.5f);//in t
				texel *= 3;
				offset = texel - (lpos.xy * 1024.f) + float2(0.5f, 0.5f);//in t

				[unroll]
				for (uint i = 0; i < 36; i++)
				{
					int2 texeloffset = int2(i / 6 - 3, i % 6 - 3);//in t
					zpos = CSM.Load(int4(texel, 0, 0), texeloffset);//in 0 to 1
					distancez = lpos.z - zpos;//in 0 to 1

					if (distancez > 0)
					{
						result += shadow(offset, texeloffset, scalexy, scalez, distancez);
					}
				}
			}
			else if (distancez > 0)
			{
				result += shadow(offset, int2(0, 0), scalexy, scalez, distancez);
			}
		}
		else if (distancez > 0)
		{
			result += shadow(offset, int2(0, 0), scalexy, scalez, distancez);
		}
	}
	else if (distancez > 0)
	{
		result += shadow(offset, int2(0, 0), scalexy, scalez, distancez);
	}

	Output[Texel] = 1 - result;
}

[numthreads(NUMTHREADS_X, NUMTHREADS_Y, 1)]
void blur(uint3 dispatchThreadID : SV_DispatchThreadID) {

	float2 Texel = dispatchThreadID.xy;

#ifndef MS_SAMPLE_COUNT
		uint c_id = cascade_id_stencil(Stencil[Texel].y);
#else
		uint c_id = cascade_id_stencil(StencilMS.Load(Texel, 0).y);
#endif

	float result;
	[branch]
	if (c_id > 1) {
		result = 0;

		for (int i = -5; i<5; i++) {
#ifdef VERTICAL
			float sample = Shadow[Texel + float2(0, i)];
#else
			float sample = Shadow[Texel + float2(i, 0)];
#endif

			result += sample * gaussian_weigth(i, 1.5);
		}
#ifdef VERTICAL
		result = pow(result, 2);
#endif		
	}
	else
	{
		result = Shadow[Texel];
	}

	Output[Texel] = result;
}