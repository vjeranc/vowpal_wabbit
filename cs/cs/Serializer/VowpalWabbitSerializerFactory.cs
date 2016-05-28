﻿// --------------------------------------------------------------------------------------------------------------------
// <copyright file="VowpalWabbitSerializerFactory.cs">
//   Copyright (c) by respective owners including Yahoo!, Microsoft, and
//   individual contributors. All rights reserved.  Released under a BSD
//   license as described in the file LICENSE.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Reflection.Emit;
using VW.Serializer.Attributes;
using VW.Serializer.Intermediate;
using VW.Labels;
using System.IO;
using System.Runtime.CompilerServices;
using System.Diagnostics.Contracts;
using VW.Reflection;

namespace VW.Serializer
{
    /// <summary>
    /// Factory to ease creation of serializers.
    /// </summary>
    public static class VowpalWabbitSerializerFactory
    {
        /// <summary>
        /// Example and example result type based serializer cache.
        /// </summary>
        private static readonly Dictionary<Type, object> SerializerCache = new Dictionary<Type, object>();

        /// <summary>
        /// Creates a serializer for the given type and settings.
        /// </summary>
        /// <typeparam name="TExample">The user type to serialize.</typeparam>
        /// <param name="settings"></param>
        /// <returns></returns>
        public static IVowpalWabbitSerializerCompiler<TExample> CreateSerializer<TExample>(VowpalWabbitSettings settings = null)
        {
            Schema schema = null;

            Type cacheKey = null;
            if (settings != null && settings.Schema != null)
            {
                schema = settings.Schema;
            }
            else
            {
                // only cache non-string generating serializer
                if (!settings.EnableStringExampleGeneration)
                {
                    cacheKey = typeof(TExample);
                    object serializer;

                    if (SerializerCache.TryGetValue(cacheKey, out serializer))
                    {
                        return (IVowpalWabbitSerializerCompiler<TExample>)serializer;
                    }
                }

                ITypeInspector typeInspector = settings.TypeInspector;
                if (typeInspector == null)
                    typeInspector = TypeInspector.Default;

                // TODO: enhance caching based on feature list & featurizer set
                // if no feature mapping is provided, use [Feature] annotation on provided type.
                schema = typeInspector.CreateSchema(settings, typeof(TExample));

                var multiExampleSerializerCompiler = VowpalWabbitMultiExampleSerializerCompiler.TryCreate<TExample>(settings, schema);
                if (multiExampleSerializerCompiler != null)
                    return multiExampleSerializerCompiler;
            }

            // need at least a single feature to do something sensible
            if (schema == null || schema.Features.Count == 0)
            {
                return null;
            }

            var newSerializer = new VowpalWabbitSingleExampleSerializerCompiler<TExample>(
                schema,
                settings == null ? null : settings.CustomFeaturizer,
                !settings.EnableStringExampleGeneration);

            if (cacheKey != null)
            {
                SerializerCache[cacheKey] = newSerializer;
            }

            return newSerializer;
        }
    }
}
