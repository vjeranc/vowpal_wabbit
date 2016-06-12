﻿// --------------------------------------------------------------------------------------------------------------------
// <copyright file="PropertyConfiguration.cs">
//   Copyright (c) by respective owners including Yahoo!, Microsoft, and
//   individual contributors. All rights reserved.  Released under a BSD
//   license as described in the file LICENSE.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

using System;
namespace VW.Serializer
{
    /// <summary>
    /// Constants used throughout C# wrapper.
    /// </summary>
    public sealed class PropertyConfiguration
    {
        public const string FeatureIgnorePrefixDefault = "_";
        public const string TextPropertyDefault = "_text";
        public const string LabelPropertyDefault = "_label";
        public const string LabelIndexPropertyDefault = "_labelIndex";
        public const string LabelPropertyPrefixDefault = "_label_";
        public const string MultiPropertyDefault = "_multi";

        public static readonly PropertyConfiguration Default = new PropertyConfiguration();

        public PropertyConfiguration()
        {
            this.FeatureIgnorePrefix = FeatureIgnorePrefixDefault;
            this.TextProperty = TextPropertyDefault;
            this.LabelProperty = LabelPropertyDefault;
            this.MultiProperty = MultiPropertyDefault;
            this.LabelIndexProperty = LabelIndexPropertyDefault;
            this.LabelPropertyPrefix = LabelPropertyPrefixDefault;
        }

        /// <summary>
        /// JSON properties starting with underscore are ignored.
        /// </summary>
        public string FeatureIgnorePrefix { get; set; }

        /// <summary>
        /// JSON property "_text" is marshalled using <see cref="VW.Serializer.StringProcessing.Split"/>.
        /// </summary>
        public string TextProperty { get; set; }

        /// <summary>
        /// JSON property "_label" is used as label.
        /// </summary>
        public string LabelProperty { get; set; }

        /// <summary>
        /// JSON property "_labelIndex" determines the index this label is applied for multi-line examples.
        /// </summary>
        public string LabelIndexProperty { get; set; }

        /// <summary>
        /// JSON properties starting with "_label_$name" are used to specify nested properties. Has the same effect as _label: { "$name": ... }.
        /// </summary>
        public string LabelPropertyPrefix { get; set; }

        /// <summary>
        /// JSON property "_multi" is used to signal multi-line examples.
        /// </summary>
        public string MultiProperty { get; set; }

        /// <summary>
        /// True if <paramref name="property"/> is considered a special property and thus should not be skipped.
        /// </summary>
        /// <param name="property">The JSON property name.</param>
        /// <returns>True if <paramref name="property"/> is a special property, false otherwise.</returns>
        public bool IsSpecialProperty(string property)
        {
            return property.Equals(TextProperty, StringComparison.Ordinal) ||
                property.Equals(LabelProperty, StringComparison.Ordinal) ||
                property.Equals(MultiProperty, StringComparison.Ordinal) ||
                property.Equals(LabelIndexProperty, StringComparison.Ordinal) ||
                property.StartsWith(LabelPropertyPrefixDefault, StringComparison.Ordinal);
        }
    }
}
